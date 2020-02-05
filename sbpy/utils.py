import numpy as np

def create_convergence_table(labels, errors, h, title=None, filename=None):
    """
    Creates a latex convergence table.
    Parameters:
      labels: An array strings describing each grid (e.g. number of nodes).
      errors: An array of errors.
      h: An array of grid sizes.

    Optional Parameters:
      title: Table title.
      filename: Write table to file.

    Output:
      Prints tex code for the table.
    """

    errors = np.array(errors)
    h = np.array(h)

    rates = (np.log(errors[:-1]) - np.log(errors[1:]))/(np.log(h[:-1]) - np.log(h[1:]))

    N = len(errors)
    print("\\begin{tabular}{|l|l|l|}")
    print("\hline")

    if title:
        print("\multicolumn{{3}}{{|c|}}{{{}}} \\\\".format(title))

    print("\hline")
    print("& error & rate \\\\".format(errors[0]))
    print("\hline")
    print(labels[0], " & {:.4e} & - \\\\".format(errors[0]))
    for k in range(1,N):
        print(labels[k], " & {:.4e} & {:.2f} \\\\".format(errors[k], rates[k-1]))
    print("\hline")
    print("\\end{tabular}")

    if filename:
        with open(filename,'a') as f:
            f.write("\\begin{tabular}{|l|l|l|}\n")
            f.write("\hline\n")

            if title:
                f.write("\multicolumn{{3}}{{|c|}}{{{}}} \\\\\n".format(title))

            f.write("\hline\n")
            f.write("& error & rate \\\\\n".format(errors[0]))
            f.write("\hline\n")
            f.write(str(labels[0]) + " & {:.4e} & - \\\\\n".format(errors[0]))
            for k in range(1,N):
                f.write(str(labels[k]) + " & {:.4e} & {:.2f} \\\\\n".format(errors[k], rates[k-1]))
            f.write("\hline\n")
            f.write("\\end{tabular}\n\n")

