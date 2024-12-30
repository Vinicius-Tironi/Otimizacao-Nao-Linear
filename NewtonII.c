#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1400
#define TOL 1e-3

double funcao(double x, double y)
{
    return -12*y + 4*pow(x,2) + 4*pow(y,2) + 4*x*y;
}

// Gradiente de f
void gradiente(double (*f)(double, double), double x, double y, double *grad_x, double *grad_y)
{
    double h = 1e-6;
    *grad_x = (f(x + h, y) - f(x, y)) / h;
    *grad_y = (f(x, y + h) - f(x, y)) / h;
}

void hessiana(double (*f)(double, double), double x, double y, double hess[2][2])
{
    double h = 1e-6;
    hess[0][0] = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h * h);
    hess[0][1] = (f(x + h, y + h) - f(x + h, y) - f(x, y + h) + f(x, y)) / (h * h);
    hess[1][0] = hess[0][1];
    hess[1][1] = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h * h);
}

// Regra de Armijo
double armijo(double (*f)(double, double), double x, double y, double alpha, double c, double beta, double grad_x, double grad_y)
{
    while (f(x - alpha * grad_x, y - alpha * grad_y) > f(x, y) - c * alpha * (grad_x * grad_x + grad_y * grad_y))
    {
        alpha *= beta;
    }
    return alpha;
}

// Método de Newton
void metodoNewton(double (*f)(double, double), double x0, double y0, double tol, int max_iter, double alpha, double c, double beta, double *f_minimo)
{
    double x = x0, y = y0;
    double grad_x, grad_y;
    double hess[2][2];

    FILE *file = fopen("INf3_Newton_Ar.txt", "w");
    fprintf(file, "# Iteração\tf(x)\t||gradiente||\n");

    gradiente(f, x, y, &grad_x, &grad_y);
    double grad_norm = sqrt(grad_x * grad_x + grad_y * grad_y);
    fprintf(file, "0\t%.8f\t%.8f\n", f(x, y), grad_norm);
    printf("Iteração 0 (início da busca): (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", x, y, f(x, y));

    for (int iter = 1; iter <= max_iter; iter++)
    {
        gradiente(f, x, y, &grad_x, &grad_y);
        hessiana(f, x, y, hess);

        alpha = armijo(f, x, y, alpha, c, beta, grad_x, grad_y);

        double det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
        if (fabs(det) < 1e-6)
        {
            printf("Determinante próximo de zero. Método de Newton pode não convergir.\n");
            break;
        }

        double prev_x = x;
        double prev_y = y;

        x -= alpha * (hess[1][1] * grad_x - hess[0][1] * grad_y) / det;
        y -= alpha * (-hess[1][0] * grad_x + hess[0][0] * grad_y) / det;

        gradiente(f, x, y, &grad_x, &grad_y);
        grad_norm = sqrt(grad_x * grad_x + grad_y * grad_y);
        double f_current = f(x, y);

        fprintf(file, "%d\t%.8f\t%.8f\n", iter, f_current, grad_norm);
        printf("Iteração %d: (x, y) = (%.6f, %.6f), f(x, y) = %.6f, ||grad|| = %.6f, alpha = %.6f\n", iter, x, y, f_current, grad_norm, alpha);

        // Critério de parada
        double dx = fabs(x - prev_x);
        double dy = fabs(y - prev_y);
        if (dx < tol && dy < tol)
        {
            printf("Critério de parada alcançado na iteração %d.\n", iter);
            break;
        }
    }

    fclose(file);
    *f_minimo = f(x, y);
    printf("Mínimo encontrado em (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", x, y, *f_minimo);
}



int main()
{
    setlocale(LC_ALL, "Portuguese");
    setlocale(LC_NUMERIC, "C");

    double x0 = 1;
    double y0 = 0;
    double alpha = 0.1;
    double c = 0.1;
    double beta = 0.5;
    double f_minimo;

    metodoNewton(funcao, x0, y0, TOL, MAX_ITER, alpha, c, beta, &f_minimo);

    FILE *gnuplot_cv = popen("gnuplot -persistent", "w");
        fprintf(gnuplot_cv, "set title 'Análise de Convergência do Método de Newton'\n");
        fprintf(gnuplot_cv, "set xlabel 'Iteração'\n");
        fprintf(gnuplot_cv, "set ylabel 'Valor'\n");
        fprintf(gnuplot_cv, "set key top right\n");
        fprintf(gnuplot_cv, "set grid\n");
        fprintf(gnuplot_cv, "set style line 1 lt 2 lc rgb 'red' lw 2 dashtype 2\n"); // Linha tracejada
        fprintf(gnuplot_cv, "set style line 2 lt 2 lc rgb 'black' lw 2 dashtype 2\n"); // Linha tracejada
        fprintf(gnuplot_cv, "plot 'INf3_Newton_Ar.txt' using 1:2 title 'f(x)' with lines lw 2, '' using 1:3 title '||gradiente||' with lines lw 2 ,");
        fprintf(gnuplot_cv, "0 with lines ls 2 title 'y = 0', ");
        fprintf(gnuplot_cv, "%f with lines ls 1 title 'y = mínimo'\n", f_minimo);

        fflush(gnuplot_cv);
        pclose(gnuplot_cv);

    return 0;
}
