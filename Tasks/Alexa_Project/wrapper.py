def add_wrapping(item):
    def wrapped_item():
        return "a wrapped up box of {}".format(str(item()))
    return wrapped_item

@add_wrapping
def new_gpu():
    return "a new Testla P100 GPU"

a = new_gpu()
print(a)