#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-6

double funcao(double x, double y)
{
    return 5 * pow(x, 2) + 5 * pow(y, 2) - x * y - 11 * x + 11 * y + 11;
}

void gradiente(double (*f)(double, double), double x, double y, double grad[])
{
    double h = 1e-6;
    grad[0] = (f(x + h, y) - f(x, y)) / h;
    grad[1] = (f(x, y + h) - f(x, y)) / h;
}

// Armijo
double armijo(double (*f)(double, double), double x, double y, double grad[], double alpha, double beta)
{
    double f_curr = f(x, y), f_new;

    while (1)
    {
        double new_x = x - alpha * grad[0];
        double new_y = y - alpha * grad[1];
        f_new = f(new_x, new_y);

        if (f_new <= f_curr - beta * alpha * (grad[0] * grad[0] + grad[1] * grad[1])) break;
        alpha *= 0.5;
    }
    return alpha;
}

// Seção Áurea
double secao_aurea(double (*f)(double, double), double x, double y, double grad[])
{
    double a = 0.0, b = 1.0;
    double phi = (sqrt(5) - 1) / 2;
    double tol = 1e-6, x1, x2;

    x1 = b - phi * (b - a);
    x2 = a + phi * (b - a);

    while ((b - a) > tol)
    {
        double f_x1 = f(x - x1 * grad[0], y - x1 * grad[1]);
        double f_x2 = f(x - x2 * grad[0], y - x2 * grad[1]);

        if (f_x1 < f_x2)
        {
            b = x2;
            x2 = x1;
            x1 = b - phi * (b - a);
        } else
        {
            a = x1;
            x1 = x2;
            x2 = a + phi * (b - a);
        }
    }
    return (a + b) / 2;
}

// Gradiente
void metodo_gradiente(double (*f)(double, double), double x, double y, double tol, int max_iter, double alpha, double beta, int metodo)
{
    double grad[2], fx_values[MAX_ITER], gradient_norms[MAX_ITER];
    int iter = 0;

    gradiente(f, x, y, grad);
    fx_values[0] = f(x, y);
    gradient_norms[0] = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

    printf("Método %s\n", metodo == 1 ? "Seção Áurea" : "Regra de Armijo");
    printf("Iteração 0: x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f\n", x, y, fx_values[0], gradient_norms[0]);

    while (iter < max_iter)
    {
        gradiente(f, x, y, grad);
        double grad_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

        double prev_x = x;
        double prev_y = y;
        double prev_f = f(x, y);

        double step = metodo == 1 ? secao_aurea(f, x, y, grad) : armijo(f, x, y, grad, alpha, beta);

        x -= step * grad[0];
        y -= step * grad[1];
        double curr_f = f(x, y);

        printf("Iteração %d: x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f, passo = %.6f\n", iter + 1, x, y, curr_f, grad_norm, step);

        // Critério de parada
        if (grad_norm < tol || fabs(x - prev_x) < tol || fabs(y - prev_y) < tol || fabs(curr_f - prev_f) < tol) break;

        fx_values[iter] = curr_f;
        gradient_norms[iter] = grad_norm;
        iter++;
    }

    printf("Mínimo encontrado: x = %.6f, y = %.6f, f(x, y) = %.6f\n\n", x, y, f(x, y));

    FILE *file = fopen(metodo == 1 ? "Grad_Secao_Aurea.txt" : "Grad_Armijo.txt", "w");
    if (file)
    {
        fprintf(file, "# Iteração\tf(x, y)\t||gradiente||\n");
        for (int i = 0; i <= iter; i++)
        {
            fprintf(file, "%d\t%.8f\t%.8f\n", i + 1, fx_values[i], gradient_norms[i]);
        }
        fclose(file);
    }
}

int main()
{
setlocale(LC_ALL, "Portuguese");
setlocale(LC_NUMERIC, "C");

    double x = 1.0, y = 1.0;
    double alpha = 0.1, beta = 0.1, tol = 1e-6;
    int max_iter = 1000;

    metodo_gradiente(funcao, x, y, tol, max_iter, alpha, beta, 1);
    metodo_gradiente(funcao, x, y, tol, max_iter, alpha, beta, 2);

    FILE *gnuplot = popen("gnuplot -persistent", "w");
        fprintf(gnuplot, "set title 'Análise de Convergência'\n");
        fprintf(gnuplot, "set xlabel 'Iteração'\n");
        fprintf(gnuplot, "set ylabel 'Valor'\n");
        fprintf(gnuplot, "set grid\n");
        fprintf(gnuplot, "plot 'Grad_Secao_Aurea.txt' using 1:2 title 'f(x,y) Seção Áurea' with lines, ");
        fprintf(gnuplot, "'Grad_Secao_Aurea.txt' using 1:3 title '||grad|| Seção Áurea' with lines, ");
        fprintf(gnuplot, "'Grad_Armijo.txt' using 1:2 title 'f(x,y) Armijo' with lines, ");
        fprintf(gnuplot, "'Grad_Armijo.txt' using 1:3 title '||grad|| Armijo' with lines\n");
        fflush(gnuplot);
        pclose(gnuplot);

    return 0;
}
