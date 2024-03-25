from mouselib import mouselib

mouse_model = mouselib("server", "0.0.0.0", 5051, None, None)
mouse_model.start()
mouse_model.join()