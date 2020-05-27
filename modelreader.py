import pickle


with open("model.md", "rb") as f:
    model = pickle.load(f)

    for v in model:
        print("Position: {}\nField Vector: {}\n".format(v, model[v]))
