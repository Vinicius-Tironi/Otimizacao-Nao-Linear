#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-6

double funcao(double x, double y) {
    return 5*pow(x, 2) + 5*pow(y, 2) - x*y - 11*x + 11*y + 11;
}

void gradiente(double (*f)(double, double), double x, double y, double grad[])
{
    double h = 1e-6;
    grad[0] = (f(x + h, y) - f(x, y)) / h;
    grad[1] = (f(x, y + h) - f(x, y)) / h;
}

double secao_aurea(double (*f)(double, double), double x, double y, double grad[])
{
    double a = 0.0, b = 1.0;
    double phi = (sqrt(5) - 1) / 2;
    double tol = 1e-6;
    double x1, x2;

    x1 = b - phi * (b - a);
    x2 = a + phi * (b - a);

    while ((b - a) > tol) {
        double f_x1 = f(x - x1 * grad[0], y - x1 * grad[1]);
        double f_x2 = f(x - x2 * grad[0], y - x2 * grad[1]);

        if (f_x1 < f_x2) {
            b = x2;
            x2 = x1;
            x1 = b - phi * (b - a);
        } else {
            a = x1;
            x1 = x2;
            x2 = a + phi * (b - a);
        }
    }

    return (a + b) / 2;
}

void metodo_gradiente(double (*f)(double, double), double x0, double y0, double tol, int max_iter) {
    double x = x0, y = y0;
    double grad[2];
    int iter = 0;

    double fx_values[MAX_ITER];
    double gradient_norms[MAX_ITER];

    gradiente(f, x, y, grad);
    fx_values[0] = f(x, y);
    gradient_norms[0] = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);        // iteração 0
    printf("Iteração 0 (início da busca): x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f\n", x, y, fx_values[0], gradient_norms[0]);


    while (iter < max_iter)
        {
        gradiente(f, x, y, grad);
        double grad_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

        fx_values[iter] = f(x, y);
        gradient_norms[iter] = grad_norm;

        if (grad_norm < tol) break;

        double alpha = secao_aurea(f, x, y, grad);

        double new_x = x - alpha * grad[0];
        double new_y = y - alpha * grad[1];
        double dx = fabs(new_x - x);
        double dy = fabs(new_y - y);

        x = new_x;
        y = new_y;

        printf("Iteração %d: x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f, alpha = %.6f\n", iter + 1, x, y, f(x, y), grad_norm, alpha);

        if (dx < tol && dy < tol) break;

        iter++;
    }

    printf("Mínimo encontrado pelo método do gradiente em x = %.6f, y = %.6f, f(x, y) = %.6f\n", x, y, f(x, y));

    // análise de convergência
    FILE *file = fopen("INf3_Grad_phi.txt", "w");
    if (file)
    {
        fprintf(file, "# Iteração\tf(x, y)\t||gradiente||\n");
        for (int i = 0; i < iter; i++)
        {
            fprintf(file, "%d\t%.8f\t%.8f\n", i + 1, fx_values[i], gradient_norms[i]);
        }
        fclose(file);
    }

    // plot da análise de convergência
    FILE *gnuplot_cv = popen("gnuplot -persistent", "w");
    if (gnuplot_cv)
    {
        fprintf(gnuplot_cv, "set title 'Análise de Convergência do Método do Gradiente'\n");
        fprintf(gnuplot_cv, "set xlabel 'Iteração'\n");
        fprintf(gnuplot_cv, "set ylabel 'Valor'\n");
        fprintf(gnuplot_cv, "set key top right\n");
        fprintf(gnuplot_cv, "set grid\n");

        double f_minimo = funcao(x, y);

        fprintf(gnuplot_cv, "plot 'INf3_Grad_phi.txt' using 1:2 title 'f(x, y)' with lines lw 2, '' using 1:3 title '||gradiente||' with lines lw 2, ");
        fprintf(gnuplot_cv, "0 with lines lt 2 lc rgb 'red' title 'y = 0', ");
        fprintf(gnuplot_cv, "%f with lines lt 2 lc rgb 'black' title 'y = mínimo'\n", f_minimo);

        fflush(gnuplot_cv);
        pclose(gnuplot_cv);
    }
}

int main()
{
    setlocale(LC_ALL, "Portuguese");
    setlocale(LC_NUMERIC, "C");
    double x0 = 1.0, y0 = 1.0;
    double tol = TOL;

    metodo_gradiente(funcao, x0, y0, tol, MAX_ITER);

    return 0;
}



