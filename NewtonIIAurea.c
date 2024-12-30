#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-3

double funcao(double x, double y)
{
    return -12*y + 4*pow(x,2) + 4*pow(y,2) + 4*x*y;
}

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

// Método de Newton
void metodoNewton(double (*f)(double, double), double x0, double y0, double tol, int max_iter, double *f_minimo)
{
    double x = x0, y = y0;
    double grad_x, grad_y;
    double hess[2][2];

    double fx_values[MAX_ITER];
    double gradient_norms[MAX_ITER];

    gradiente(f, x, y, &grad_x, &grad_y);
    fx_values[0] = f(x, y);
    gradient_norms[0] = sqrt(grad_x * grad_x + grad_y * grad_y);

    printf("Iteração 0 (início da busca): (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", x, y, f(x, y));

    FILE *file = fopen("INf3_Newton_phi.txt", "w");
    if (file)
    {
        fprintf(file, "# Iteração\tf(x)\t||gradiente||\n");
        fprintf(file, "0\t%.8f\t%.8f\n", fx_values[0], gradient_norms[0]);
    }

    for (int iter = 0; iter < max_iter; iter++)
    {
        double prev_x = x;
        double prev_y = y;

        gradiente(f, x, y, &grad_x, &grad_y);
        hessiana(f, x, y, hess);

        double det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
        if (fabs(det) < 1e-6)
        {
            printf("Determinante próximo de zero. Método de Newton pode não convergir.\n");
            break;
        }

        double d1 = -(hess[1][1] * grad_x - hess[0][1] * grad_y) / det;
        double d2 = -(-hess[1][0] * grad_x + hess[0][0] * grad_y) / det;

        double alpha = secaoAurea(f, x, y, d1, d2, tol);

        x += alpha * d1;
        y += alpha * d2;

        fx_values[iter + 1] = f(x, y);
        gradient_norms[iter + 1] = sqrt(grad_x * grad_x + grad_y * grad_y);

        printf("Iteração %d: (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", iter + 1, x, y, f(x, y));

        double dx = fabs(x - prev_x);
        double dy = fabs(y - prev_y);

        // Critério de parada
        if (dx < tol && dy < tol)
        {
            printf("Critério de parada alcançado na iteração %d.\n", iter + 1);
            break;
        }

        fprintf(file, "%d\t%.8f\t%.8f\n", iter + 1, fx_values[iter + 1], gradient_norms[iter + 1]);
    }

    fclose(file);

    *f_minimo = f(x, y);
    printf("Mínimo encontrado em (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", x, y, f(x, y));
}

int main()
{
setlocale(LC_ALL, "Portuguese");
setlocale(LC_NUMERIC, "C");

    double x0 = 1;
    double y0 = 0;
    double f_minimo;

    metodoNewton(funcao, x0, y0, TOL, MAX_ITER, &f_minimo);

    FILE *gnuplot_cv = popen("gnuplot -persistent", "w");
        fprintf(gnuplot_cv, "set title 'Análise de Convergência do Método de Newton'\n");
        fprintf(gnuplot_cv, "set xlabel 'Iteração'\n");
        fprintf(gnuplot_cv, "set ylabel 'Valor'\n");
        fprintf(gnuplot_cv, "set key top right\n");
        fprintf(gnuplot_cv, "set grid\n");
        fprintf(gnuplot_cv, "set style line 1 lt 2 lc rgb 'red' lw 2 dashtype 2\n");
        fprintf(gnuplot_cv, "set style line 2 lt 2 lc rgb 'black' lw 2 dashtype 2\n");
        fprintf(gnuplot_cv, "plot 'INf3_Newton_phi.txt' using 1:2 title 'f(x)' with lines lw 2, '' using 1:3 title '||gradiente||' with lines lw 2 ,");
        fprintf(gnuplot_cv, "0 with lines ls 2 title 'y = 0', ");
        fprintf(gnuplot_cv, "%f with lines ls 1 title 'y = mínimo'\n", f_minimo);

        fflush(gnuplot_cv);
        pclose(gnuplot_cv);

    return 0;
}
