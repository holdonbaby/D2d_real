# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import d2dmodel.d2d01package
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n=5
    c=10
    w=[2,2,6,5,4]
    v=[6,3,5,4,6]
    res=d2dmodel.d2d01package.bag(n,c,w,v)
    d2dmodel.d2d01package.show(n,c,w,res)
    plt.plot(w,v)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
