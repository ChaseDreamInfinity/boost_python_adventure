import py_selective_search as pyss

filename = "poodle2.jpg"
windows = pyss.get_windows(filename)

print("%d boxes in total" % len(windows))
