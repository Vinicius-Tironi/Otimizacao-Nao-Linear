#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-1

double funcao(double x, double y)
{
    return pow(x, 4) - 8*pow(x, 3) + 25*pow(x, 2) - 4*x*y - 32*x + 4*pow(y, 2) + 16;
}

void gradiente(double (*f)(double, double), double x, double y, double *grad_x, double *grad_y)
{
    double h = 1e-6;
    *grad_x = (f(x + h, y) - f(x, y)) / h;
    *grad_y = (f(x, y + h) - f(x, y)) / h;
}

// Seção Áurea
double secaoAurea(double (*f)(double, double), double x, double y, double d1, double d2, double tol)
{
    const double phi = (1 + sqrt(5)) / 2;
    double a = -1.0, b = 1.0;
    while (fabs(b - a) > tol) {
        double x1 = b - (b - a) / phi;
        double x2 = a + (b - a) / phi;
        if (f(x + d1 * x1, y + d2 * x1) < f(x + d1 * x2, y + d2 * x2))
        {
            b = x2;
        } else
        {
            a = x1;
        }
    }
    return (a + b) / 2;
}

// Quasi-Newton BFGS
void metodoBFGS(double (*f)(double, double), double x0, double y0, double tol, int max_iter, double *f_minimo)
{
    double x = x0, y = y0;
    double grad_x, grad_y, grad_x_ant, grad_y_ant;
    double H[2][2] = {{1, 0}, {0, 1}};
    gradiente(f, x, y, &grad_x, &grad_y);

    FILE *file = fopen("f5_NewtonBFGS_phi.txt", "w");
    fprintf(file, "# Iteração\tf(x)\t||gradiente||\n");

    int iter;
    for (iter = 0; iter < max_iter; iter++)
        {
        double grad_norm = sqrt(grad_x * grad_x + grad_y * grad_y);

        printf("Iter %d: x = %.6f, y = %.6f, f(x, y) = %.6f, ||gradiente|| = %.6f\n", iter, x, y, f(x, y), grad_norm);
        fprintf(file, "%d\t%.8f\t%.8f\n", iter, f(x, y), grad_norm);

        // Critério de Convergência
        if (grad_norm < tol)
        {
            printf("Convergência alcançada na iteração %d.\n", iter);
            break;
        }

        double d1 = -(H[0][0] * grad_x + H[0][1] * grad_y);
        double d2 = -(H[1][0] * grad_x + H[1][1] * grad_y);
        double alpha = secaoAurea(f, x, y, d1, d2, tol);

        double x_ant = x, y_ant = y;
        x += alpha * d1;
        y += alpha * d2;

        grad_x_ant = grad_x;
        grad_y_ant = grad_y;
        gradiente(f, x, y, &grad_x, &grad_y);

        double s1 = x - x_ant, s2 = y - y_ant;
        double y1 = grad_x - grad_x_ant, y2 = grad_y - grad_y_ant;
        double ys = y1 * s1 + y2 * s2;

        if (fabs(ys) < 1e-8)
        {
            H[0][0] = 1; H[0][1] = 0; H[1][0] = 0; H[1][1] = 1;
            printf("Recondicionamento de H na iteração %d.\n", iter); // Recondicionamento utiliza matriz identidade
            continue;
        }

        double rho = 1.0 / ys;
        H[0][0] += rho * (s1 * s1);
        H[0][1] += rho * (s1 * s2);
        H[1][0] += rho * (s2 * s1);
        H[1][1] += rho * (s2 * s2);
    }

    fclose(file);

    *f_minimo = f(x, y);
    printf("Mínimo encontrado em (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", x, y, *f_minimo);

}

int main()
{
    setlocale(LC_ALL, "Portuguese");
    double x0 = 1.0, y0 = 1.0, f_minimo;

    metodoBFGS(funcao, x0, y0, TOL, MAX_ITER, &f_minimo);

    FILE *gnuplot = popen("gnuplot -persistent", "w");
        fprintf(gnuplot, "set title 'Convergência do Método BFGS'\n");
        fprintf(gnuplot, "set xlabel 'Iterações'\n");
        fprintf(gnuplot, "set ylabel 'Valores'\n");
        fprintf(gnuplot, "set key outside\n");
        fprintf(gnuplot, "plot 'f5_NewtonBFGS_phi.txt' using 1:2 with linespoints title 'f(x)', ");
        fprintf(gnuplot, "'f5_NewtonBFGS_phi.txt' using 1:3 with linespoints title '||gradiente||'\n");
        pclose(gnuplot);

    return 0;
}
