import pickle


with open("model.md", "rb") as f:
    model = pickle.load(f)

    with open("output.txt", "w") as f2:

        for v in model:
            f2.write("Position: {}\nField Vector: {}\n\n".format(v, model[v]))
