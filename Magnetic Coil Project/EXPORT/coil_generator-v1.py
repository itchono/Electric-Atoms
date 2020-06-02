import math

def x(t):
    return 0.05*t
    

def y(t):
    return 2*math.sin(t)

def z(t):
    return 2*math.cos(t)

def I(t):
    return 1


if __name__ == "__main__":
    with open("testcoil.txt", "w") as f:
        for t in range(0, 3000):
            print(t)
            f.write("{},{},{},{}\n".format(x(t/10), y(t/10), z(t/10), I(t/10)))
