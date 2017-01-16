#ifndef SELECTIVE_SEARCH_H
#define SELECTIVE_SEARCH_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <math.h>
#include <iostream>
#include "segment-image.h"

#define KERNELSIZE 5
#define KERNELSIGMA 1
#define PI 3.14159265
#define RESIZE_IMAGE_WIDTH 500


enum ColorSpace {
	BGR,
	HSV,
	Lab
};

enum SimFunction {
	ColorTextureSizeFill,
	TextureSizeFill
};

using namespace cv;
using namespace std;

static int merge_region(Mat *nbr_metric_p, Mat *bbox_p, int merge_row_idx) {
	Mat nbr_metric_new;
	vector<int> new_nbr_idx;
	int idx0, idx1;
	int idx_a = (int)nbr_metric_p->at<float>(merge_row_idx, 0);
	int idx_b = (int)nbr_metric_p->at<float>(merge_row_idx, 1);
	int idx_new = bbox_p->rows;

	// Add the merged bounding box
	Mat new_bbox_item = Mat::zeros(1, bbox_p->cols, CV_32S); // [xmin, ymin, xmax, ymax, region size]
	new_bbox_item.at<int>(0, 0) = min(bbox_p->at<int>(idx_a, 0), bbox_p->at<int>(idx_b, 0));
	new_bbox_item.at<int>(0, 1) = min(bbox_p->at<int>(idx_a, 1), bbox_p->at<int>(idx_b, 1));
	new_bbox_item.at<int>(0, 2) = max(bbox_p->at<int>(idx_a, 2), bbox_p->at<int>(idx_b, 2));
	new_bbox_item.at<int>(0, 3) = max(bbox_p->at<int>(idx_a, 3), bbox_p->at<int>(idx_b, 3));
	new_bbox_item.at<int>(0, 4) = bbox_p->at<int>(idx_a, 4) + bbox_p->at<int>(idx_b, 4);
	bbox_p->push_back(new_bbox_item);

	for (int y = 0; y < nbr_metric_p->rows; y++) {
		idx0 = (int)nbr_metric_p->at<float>(y, 0);
		idx1 = (int)nbr_metric_p->at<float>(y, 1);

		if (idx0 != idx_a && idx0 != idx_b) {
			if (idx1 != idx_a && idx1 != idx_b) {
				nbr_metric_new.push_back(nbr_metric_p->row(y));
			}
			else {
				if (find(new_nbr_idx.begin(), new_nbr_idx.end(), idx0) == new_nbr_idx.end()) {
					new_nbr_idx.push_back(idx0);
				}
			}
		}
		else if (idx1 != idx_a && idx1 != idx_b) {
			if (find(new_nbr_idx.begin(), new_nbr_idx.end(), idx1) == new_nbr_idx.end()) {
				new_nbr_idx.push_back(idx1);
			}
		}
	}


	for (int y = 0; y < new_nbr_idx.size(); y++) {
		Mat tmp = Mat::zeros(1, nbr_metric_p->cols, CV_32F);
		tmp.at<float>(0, 0) = (float)idx_new;
		tmp.at<float>(0, 1) = (float)new_nbr_idx[y];
		nbr_metric_new.push_back(tmp);
	}

	nbr_metric_p->release();
	*nbr_metric_p = nbr_metric_new;

	return (int)new_nbr_idx.size();
}

static float cal_sim_texture_color(const Mat &A, const Mat &B) {
	return (float)(sum(min(A, B))[0]);
}

static float cal_sim_size(int size_a, int size_b, int size_im) {
	float sim_size = 1 - ((float)(size_a + size_b)) / (float)size_im;
	return sim_size;
}

static float cal_sim_fill(Mat *bbox_p, int a, int b, int size_im) {
	int xmin, ymin, xmax, ymax, size_bb;
	float sim_fill;

	xmin = min(bbox_p->at<int>(a, 0), bbox_p->at<int>(b, 0));
	ymin = min(bbox_p->at<int>(a, 1), bbox_p->at<int>(b, 1));
	xmax = min(bbox_p->at<int>(a, 2), bbox_p->at<int>(b, 2));
	ymax = min(bbox_p->at<int>(a, 3), bbox_p->at<int>(b, 3));
	size_bb = (xmax - xmin + 1) * (ymax - ymin + 1);

	sim_fill = 1 - ((float)(size_bb - bbox_p->at<int>(a, 4) - bbox_p->at<int>(b, 4))) / ((float)size_im);

	return sim_fill;
}


static Mat filter_box_length(Mat src, int min_box_len) {
	Mat des;
	int width, height;

	for (int i = 0; i < src.rows; ++i) {
		width = src.at<int>(i, 2) - src.at<int>(i, 0) + 1;
		height = src.at<int>(i, 3) - src.at<int>(i, 1) + 1;
		if (width > min_box_len && height > min_box_len) {
			des.push_back(src.row(i));
		}
	}
	return des;
}

static Mat get_unique_row(Mat src) {
	Mat des;

	// keep only unique rows
	des.push_back(src.row(0));
	for (int i = 1; i < src.rows; ++i) {
		int isInside = false;
		for (int j = 0; j < des.rows; ++j) {
			int count = 0;
			for (int k = 0; k < des.cols; ++k) 
				if (src.at<int>(i, k) == des.at<int>(j, k))
					++count;
			if (count == des.cols) {
				isInside = true;
				break;
			}
		}
		if (isInside == false) des.push_back(src.row(i));
	}
	return des;
}


static float cal_sim_total(const Mat &nbr_metric_row, SimFunction sim_func) {
	float sim_texture = nbr_metric_row.at<float>(0, 2);
	float sim_size = nbr_metric_row.at<float>(0, 3);
	float sim_fill = nbr_metric_row.at<float>(0, 4);
	float sim_color = nbr_metric_row.at<float>(0, 5);
	
	float sim_weight[4] = {1., 1., 1., 1.};
	switch (sim_func) {
		case ColorTextureSizeFill:
			break;
		case TextureSizeFill:
			sim_weight[3] = (float) 1e-8;
			break;
		default:
			break;
	}

	float sim_total = (float)(sim_weight[0] * sim_texture + sim_weight[1] * sim_size
		+ sim_weight[2] * sim_fill + sim_weight[3] * sim_color);

	return sim_total;
}

static Mat selective_search_fn(Mat img_org, const ColorSpace color_space, const int seg_threshold, const SimFunction sim_func) {
	
	Mat img;
	float hsv_hscale;

	const int nChannels = img_org.channels();
	const int nRows = img_org.rows;
	const int nCols = img_org.cols;
	const int nSize = nRows * nCols;

	switch (color_space) {
		case BGR:
			img = img_org.clone();
			break;
		case HSV:
			hsv_hscale = (float)(255. / 180.);
			cvtColor(img_org, img, CV_BGR2HSV);
			for (int x = 0; x < nCols; x++) {
				for (int y = 0; y < nRows; y++) {
					img.at<Vec3b>(y, x)[0] = (uchar)((float)img.at<Vec3b>(y, x)[0] * hsv_hscale);
				}
			}
			break;
		case Lab:
			cvtColor(img_org, img, CV_BGR2Lab);

		default:
			break;
	}

	// Convert the image format from 'Mat' to image<rgb>
	image<rgb>* theIm = new image<rgb>(nCols, nRows);
	for (int x = 0; x < nCols; x++) {
		for (int y = 0; y < nRows; y++) {
			imRef(theIm, x, y).r = img.at<Vec3b>(y, x)[2];
			imRef(theIm, x, y).g = img.at<Vec3b>(y, x)[1];
			imRef(theIm, x, y).b = img.at<Vec3b>(y, x)[0];
		}
	}

	// Gaussian derivative filtering
	Mat kernelx = Mat::zeros(KERNELSIZE, KERNELSIZE, CV_32F);
	Mat kernely;
	float tmp_kernel_exp_mul = (float)KERNELSIGMA;
	const int halfKernelSize = ((int)KERNELSIZE) / 2;
	tmp_kernel_exp_mul = (float)(1. / (tmp_kernel_exp_mul*tmp_kernel_exp_mul * 2));
	for (int x = -halfKernelSize; x <= halfKernelSize; x++) {
		for (int y = -halfKernelSize; y <= halfKernelSize; y++) {
			kernelx.at<float>(y + halfKernelSize, x + halfKernelSize) = -x * expf(-(x * x + y * y) * tmp_kernel_exp_mul);
		}
	}
	kernelx *= (2. / (float)(sum(abs(kernelx))[0])); // normalization
	transpose(kernelx, kernely);

	// Convolution
	Mat img_gradx, img_grady;
	filter2D(img, img_gradx, CV_32F, kernelx);
	filter2D(img, img_grady, CV_32F, kernely);

	// Get gradient orientation and magnitude
	Mat img_magnitude = Mat::zeros(nRows, nCols, CV_32FC3);
	Mat img_orientation = Mat::zeros(nRows, nCols, CV_8UC3); // [0,1,2,3,4,5,6,7] => [left, left-bottom, ..., right, ..., left-top] (counter-clockwise)
	float tmp_orientation_bin;
	int num_orientation_bin = 8;
	int num_half_orientation_bin = num_orientation_bin / 2;
	for (int x = 0; x < nCols; x++) {
		for (int y = 0; y < nRows; y++) {
			for (uchar z = 0; z < nChannels; z++) {
				img_magnitude.at<Vec3f>(y, x)[z] = sqrt(img_gradx.at<Vec3f>(y, x)[z] * img_gradx.at<Vec3f>(y, x)[z] + img_grady.at<Vec3f>(y, x)[z] * img_grady.at<Vec3f>(y, x)[z]);
				tmp_orientation_bin = (float)(atan2(img_grady.at<Vec3f>(y, x)[z], img_gradx.at<Vec3f>(y, x)[z]) * num_half_orientation_bin / PI);
				img_orientation.at<Vec3b>(y, x)[z] = ((uchar)round(tmp_orientation_bin) + num_half_orientation_bin) % num_orientation_bin;
			}
		}
	}
	normalize(img_magnitude, img_magnitude, 0, 0.9999999999999999, NORM_MINMAX, CV_32FC3); // normalization for the magnitude

																						   // Call Felzenswalb segmentation algorithm
	int num_css;
	const float sigma = 0.8f;
	const float c = (float)seg_threshold;
	const int min_size = seg_threshold;
	int* seg_indices = segment_image_index(theIm, sigma, c, min_size, &num_css);

	// Convert back to Mat format
	Mat out_seg_ind = Mat(nCols, nRows, CV_32SC1, seg_indices);
	transpose(out_seg_ind, out_seg_ind);

	// Calculate the bounding box, neighbor indices, texture histogram, color histogram
	const int texture_bin_size = 10;
	const int color_bin_size = 25;
	Mat img_texture_hist = Mat::zeros(num_css, texture_bin_size * num_orientation_bin * nChannels, CV_32F);
	Mat img_color_hist = Mat::zeros(num_css, color_bin_size * nChannels, CV_32F);

	Mat bbox = Mat::zeros(num_css, 6, CV_32S); // [xmin, ymin, xmax, ymax, region size, priority]
	bbox.col(0).setTo(Scalar(nCols));
	bbox.col(1).setTo(Scalar(nRows));
	bbox.col(5).setTo(Scalar(num_css));
	Mat nbr_regions = Mat::eye(num_css, num_css, CV_8U); // Track neighboring regions

	int prev_ver_idx, prev_hor_idx, curr_idx, texture_hist_idx, color_hist_idx;

	for (int x = 0; x < nCols; x++) {
		prev_ver_idx = out_seg_ind.at<int>(0, x) - 1;
		for (int y = 0; y < nRows; y++) {
			curr_idx = out_seg_ind.at<int>(y, x) - 1;

			bbox.at<int>(curr_idx, 4) += 1;

			// get the bounding boxes
			if (bbox.at<int>(curr_idx, 0) > x) {
				bbox.at<int>(curr_idx, 0) = x;
			}
			if (bbox.at<int>(curr_idx, 1) > y) {
				bbox.at<int>(curr_idx, 1) = y;
			}
			if (bbox.at<int>(curr_idx, 2) < x) {
				bbox.at<int>(curr_idx, 2) = x;
			}
			if (bbox.at<int>(curr_idx, 3) < y) {
				bbox.at<int>(curr_idx, 3) = y;
			}

			// mark the neighboring blobs: vertical and horizontal
			nbr_regions.at<uchar>(curr_idx, prev_ver_idx) = 1;
			nbr_regions.at<uchar>(prev_ver_idx, curr_idx) = 1;
			if (x > 0) {
				prev_hor_idx = out_seg_ind.at<int>(y, x - 1) - 1;
				nbr_regions.at<uchar>(curr_idx, prev_hor_idx) = 1;
				nbr_regions.at<uchar>(prev_hor_idx, curr_idx) = 1;
			}

			// calculate the texture histogram per region per channel per orentation
			for (uchar z = 0; z < nChannels; z++) {
				// update texture histogram
				texture_hist_idx = texture_bin_size * num_orientation_bin * z;
				texture_hist_idx += (int)floor(img_magnitude.at<Vec3f>(y, x)[z] * texture_bin_size);
				texture_hist_idx += texture_bin_size * img_orientation.at<Vec3b>(y, x)[z];
				img_texture_hist.at<float>(curr_idx, texture_hist_idx) += 1;

				// update color histogram
				color_hist_idx = color_bin_size * z;
				color_hist_idx += (int)floor(((float)img.at<Vec3b>(y, x)[z]) / 256. * color_bin_size);
				img_color_hist.at<float>(curr_idx, color_hist_idx) += 1;
			}

			prev_ver_idx = curr_idx;
		}
	}
	delete[] seg_indices;
	img_magnitude.release();
	img_orientation.release();
	img.release();

	// Histogram L1 normalization: texture and color
	for (int y = 0; y < num_css; y++) {
		img_texture_hist.row(y) *= (1. / (float)(sum(img_texture_hist.row(y))[0]));
		img_color_hist.row(y) *= (1. / (float)(sum(img_color_hist.row(y))[0]));
	}

	// Calculate the similarity between neighbors
	int num_unique_nbr = (countNonZero(nbr_regions) - num_css) / 2;
	Mat nbr_metric = Mat::zeros(num_unique_nbr, 6, CV_32F); // [idx_a, idx_b, sim_texture, sim_size, sim_fill, sim_color]
	int nbr_pair_idx = 0;
	for (int x = 0; x < num_css; x++) {
		for (int y = x + 1; y < num_css; y++) {
			if (nbr_regions.at<uchar>(y, x) == 1) {
				nbr_metric.at<float>(nbr_pair_idx, 0) = (float)x;
				nbr_metric.at<float>(nbr_pair_idx, 1) = (float)y;
				nbr_metric.at<float>(nbr_pair_idx, 2) = cal_sim_texture_color(img_texture_hist.row(x), img_texture_hist.row(y));
				nbr_metric.at<float>(nbr_pair_idx, 3) = cal_sim_size(bbox.at<int>(x, 4), bbox.at<int>(y, 4), nSize);
				nbr_metric.at<float>(nbr_pair_idx, 4) = cal_sim_fill(&bbox, x, y, nSize);
				nbr_metric.at<float>(nbr_pair_idx, 5) = cal_sim_texture_color(img_color_hist.row(x), img_color_hist.row(y));
				nbr_pair_idx++;
			}
		}
	}

	// Hierarchical region merge
	int max_sim_idx, merge_idx_a, merge_idx_b, num_new_nbr, sim_idx_a, sim_idx_b;
	float max_sim_val, cur_sim_val;
	while (nbr_metric.rows > 0) {
		// find the most similar neighbors
		max_sim_val = 0;
		for (int y = 0; y < nbr_metric.rows; y++) {
			cur_sim_val = cal_sim_total(nbr_metric.row(y), sim_func);
			if (cur_sim_val > max_sim_val) {
				max_sim_val = cur_sim_val;
				max_sim_idx = y;
			}
		}

		// Calculate the texture and color histogram of the merged region
		merge_idx_a = (int)nbr_metric.at<float>(max_sim_idx, 0);
		merge_idx_b = (int)nbr_metric.at<float>(max_sim_idx, 1);

		Mat new_texture_hist_item = img_texture_hist.row(merge_idx_a) * (float)bbox.at<int>(merge_idx_a, 4)
			+ img_texture_hist.row(merge_idx_b) * (float)bbox.at<int>(merge_idx_b, 4);
		new_texture_hist_item *= 1. / (float)(bbox.at<int>(merge_idx_a, 4) + bbox.at<int>(merge_idx_b, 4));
		img_texture_hist.push_back(new_texture_hist_item);

		Mat new_color_hist_item = img_color_hist.row(merge_idx_a) * (float)bbox.at<int>(merge_idx_a, 4)
			+ img_color_hist.row(merge_idx_b) * (float)bbox.at<int>(merge_idx_b, 4);
		new_color_hist_item *= 1. / (float)(bbox.at<int>(merge_idx_a, 4) + bbox.at<int>(merge_idx_b, 4));
		img_color_hist.push_back(new_color_hist_item);

		// Merge the most similar neighbors
		num_new_nbr = merge_region(&nbr_metric, &bbox, max_sim_idx);

		// Calculate the similarity between the new region and its neighbors
		for (int y = nbr_metric.rows - 1; y >= (nbr_metric.rows - num_new_nbr); y--) {
			sim_idx_a = (int)nbr_metric.at<float>(y, 0);
			sim_idx_b = (int)nbr_metric.at<float>(y, 1);
			nbr_metric.at<float>(y, 2) = cal_sim_texture_color(img_texture_hist.row(sim_idx_a), img_texture_hist.row(sim_idx_b));
			nbr_metric.at<float>(y, 3) = cal_sim_size(bbox.at<int>(sim_idx_a, 4), bbox.at<int>(sim_idx_b, 4), nSize);
			nbr_metric.at<float>(y, 4) = cal_sim_fill(&bbox, sim_idx_a, sim_idx_b, nSize);
			nbr_metric.at<float>(y, 5) = cal_sim_texture_color(img_color_hist.row(sim_idx_a), img_color_hist.row(sim_idx_b));
		}
	}
	nbr_metric.release();
	img_texture_hist.release();
	img_color_hist.release();

	for (int y = num_css; y < bbox.rows; y++) {
		bbox.at<int>(y, 5) = y - num_css + 1;
	}

	return bbox;
}


Mat selective_search_rcnn(const cv::String &filename, bool is_show_boxes = false) {

	// Load an image
	const Mat img = imread(filename);

	if (img.empty())
	{
		Mat bbox;
		return bbox;
	}

	// Resize the image
	Mat img_resized;
	float scale_ins = (float)((float)img.cols / (float)RESIZE_IMAGE_WIDTH);
	Size size_ins((int)RESIZE_IMAGE_WIDTH, (int)(round((float)img.rows/scale_ins)));
	resize(img, img_resized, size_ins, 0, 0, INTER_CUBIC);

	const int nChannels = img_resized.channels();
	const int nRows = img_resized.rows;
	const int nCols = img_resized.cols;
	const int nSize = nRows * nCols;

	vector<ColorSpace> color_space_set{ HSV}; //color_space_set{ HSV, Lab };
	vector<int> seg_thresh_set{200}; // seg_thresh_set{ 50, 100, 150, 300};
	vector<SimFunction> sim_func_set{ ColorTextureSizeFill}; //sim_func_set{ ColorTextureSizeFill, TextureSizeFill };

	Mat bbox_ret;
	Mat bbox_all;
	for (vector<int>::iterator seg_thresh = seg_thresh_set.begin(); seg_thresh < seg_thresh_set.end(); seg_thresh++) {
		for (vector<ColorSpace>::iterator color_space = color_space_set.begin(); color_space < color_space_set.end(); color_space++) {
			for (vector<SimFunction>::iterator sim_func = sim_func_set.begin(); sim_func < sim_func_set.end(); sim_func++) {
				bbox_ret = selective_search_fn(img_resized, *color_space, *seg_thresh, *sim_func);
				if (bbox_all.empty()) {
					bbox_all = bbox_ret.clone();
				}
				else {
					vconcat(bbox_all, bbox_ret, bbox_all);
				}
			}
		}
	}

	// Process the priority
	Mat priority;
	bbox_all.col(bbox_all.cols - 1).convertTo(priority, CV_32F);
	Mat_<float> arr_rand(bbox_all.rows, 1);
	randu(arr_rand, Scalar(0), Scalar(1));
	priority = priority.mul(arr_rand);

	// Sort the priority
	Mat1i sort_idx;
	sortIdx(priority, sort_idx, SORT_EVERY_COLUMN + SORT_ASCENDING);
	Mat bbox(bbox_all.rows, 4, bbox_all.type());
	for (int y = 0; y < bbox_all.rows; y++) {
		bbox_all.row(y).colRange(0, 4).copyTo(bbox.row(sort_idx(y, 0)));
	}

	// Filter the bounding boxes by length and get unique bounding boxes
	int min_box_len = 20;
	bbox = filter_box_length(bbox, min_box_len);
	bbox = get_unique_row(bbox);

	// Scale back the bounding boxes
	bbox.convertTo(bbox, CV_32F);
	bbox.convertTo(bbox, CV_32S, scale_ins);

	// Filter the bounding boxes by length and get unique bounding boxes
	bbox = filter_box_length(bbox, min_box_len);
	bbox = get_unique_row(bbox);

	// Show multiple images in one window
	if (is_show_boxes) {
		RNG rng(12345);
		Mat img_box;		
		Scalar color;
        int start_i, n_box_per_disp = 10;

		cout << bbox.rows << " boxes in total"<< endl;
		
        for (int i = 0; i < bbox.rows; i++) {
			if (i % n_box_per_disp == 0) {
				img_box = img.clone();
				start_i = i;
			}
			color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));			
			rectangle(img_box, Point2f(bbox.at<int>(i, 0), bbox.at<int>(i, 1)), 
				Point2f(bbox.at<int>(i, 2), bbox.at<int>(i, 3)), color);
			putText(img_box, to_string(i),
				Point(bbox.at<int>(i, 0), bbox.at<int>(i, 1)+5), 
				FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,255));
            putText(img_box, to_string(i),
				Point(bbox.at<int>(i, 2)-5, bbox.at<int>(i, 3)), 
				FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0,0,255));
            if (((i+1) % n_box_per_disp == 0) ||
				(i == (bbox.rows-1) )) 
			{
				imshow("Boxes " + to_string(start_i) + "-" + to_string(i), img_box);			
			}
		}
	}

	return bbox;
}




#endif
