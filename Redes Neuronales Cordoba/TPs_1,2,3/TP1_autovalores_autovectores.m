c = 0;z = 0;
A = [0.1-0.02*z -0.02*z; 0.01*c -0.3+0.01*c;]
[Vectors, Values] = eig(A)

c = 30;z = 5;
A = [0.1-0.02*z -0.02*z; 0.01*c -0.3+0.01*c;]
[Vectors, Values] = eig(A)