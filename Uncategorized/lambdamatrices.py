'''
Mingde's Matrix Mashup
'''

def lambdamatrices(N):
    '''
    Generates a lambda basis for an N-level quantum system,
    returns list of <N^2-1> NxN matrices (double-lists), each
    representing some basis vector for a traceless hamiltonian matrix
    '''

    M = []
    
    # generate the non-diagonal elements
    for r in range(1, N):
        for c in range(r):

            # imaginary hermitian matrix
            m = [[0] * N for i in range(N)]
            m[r][c] = complex(0, 1)
            m[c][r] = complex(0, -1)

            # real hermitiian matrix
            n = [[0] * N for i in range(N)]
            n[r][c] = 1
            n[c][r] = 1
            
            M.append(m)
            M.append(n)

    # generate the diagonal elements
    # N-1 matrices needed; traceless, hermitian
    for i in range(N-1):
        k = [[(1 if (r == c and r != i) else (-N+1 if r == c == i else 0))
              for c in range(N)] for r in range(N)]
        # possibly the nastiest list comprehension you have ever seen
        # basically puts ones everywhere and another number to make sure
        # that the tr(k) = 0
        
        M.append(k)
        
    return M

def isHermitian(M):
    '''
    checks if matrix is hermitian
    '''
    N = len(M)

    for r in range(N):
        for c in range(N):
            if M[c][r] != complex(M[r][c].real, (-1)*M[r][c].imag): return False

    return True

def tr(M):
    '''
    Trace; sum of diagonals
    '''
    return sum([M[i][i] for i in range(len(M))])

def mult(A, B):
    '''
    Multiply 2 square matrices
    '''
    N = len(A)

    result = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] += A[i][k] * B[k][j]
    return result

def showMTX(M):
    '''
    Prints a square matrix in a somewhat tolerable format.
    '''
    # tried to emulate numpy style
    print("Matrix[{0}][{0}]".format(len(M)))
    for r in M:
        for c in r:
            print(c, end="\t")
        print()
    print()

def writeMTX(M, f):
    '''
    Writes a square matrix to file.
    '''

    f.write("Matrix[{0}][{0}]\n".format(len(M)))
    for r in M:
        for c in r:
            f.write(str(c)+"\t")
        f.write("\n")
    f.write("\n")

def isBasisOrthonormal(B):
    for M in B:
        for N in ([i for i in B if i != M]):
            if tr(mult(M, N)):
                print("Offending matrices:")
                showMTX(M)
                showMTX(N)
                return False

    return True

def isBasisHermitian(B):
    for M in B:
        if not isHermitian(M): return False

    return True


if __name__ == "__main__":
    '''
    Lets you choose some N, and will store the coorresponding matrices
    to a file. Also prints to screen if it's small enough.
    '''
    inp = ""
    while inp != "-1":

        while not inp.isnumeric():
            inp = input("N?\n")
        
        M = lambdamatrices(eval(inp))
        

        with open("{0}x{0}.txt".format(eval(inp)), "w") as f:
            for m in M: writeMTX(m,f)

        if eval(inp) < 10:
            for m in M: showMTX(m)  

        print("Hermitian? {}\nOrthonormal? {}".format(isBasisHermitian(M), isBasisOrthonormal(M)))
        inp = ""
                             
