#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-6
#define TOL_PONTOS 1e-6
#define TOL_FUNC 1e-6

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

// Seção Áurea
double secao_aurea(double (*f)(double, double), double x, double y, double grad[])
{
    double a = 0.0, b = 1.0, phi = (sqrt(5) - 1) / 2, tol = 1e-6, x1, x2;

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

// Regra de Armijo
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

// Gradiente Conjugado
void metodo_gradiente_conjugado(double (*f)(double, double), double x0, double y0, double tol, int max_iter, int metodo, const char *output_file)
{
    double x = x0, y = y0;
    double grad[2];
    int iter = 0;
    double fx_values[MAX_ITER], gradient_norms[MAX_ITER];
    double alpha = 0.1, beta = 0.1;

    FILE *file = fopen(output_file, "w");
    fprintf(file, "# Iteracao\tf(x, y)\t||gradiente||\n");

    printf("\nMétodo %s\n", metodo == 1 ? "Seção Áurea" : "Regra de Armijo");
    printf("Iteração %d: x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f\n", 0, x, y, f(x, y), 0.0);

    while (iter < max_iter)
    {
        gradiente(f, x, y, grad);
        double grad_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
        double step = metodo == 1 ? secao_aurea(f, x, y, grad) : armijo(f, x, y, grad, alpha, beta);

        double new_x = x - step * grad[0];
        double new_y = y - step * grad[1];
        double dx = fabs(new_x - x);
        double dy = fabs(new_y - y);
        double delta_f = fabs(f(new_x, new_y) - f(x, y));

        x = new_x;
        y = new_y;
        fx_values[iter] = f(x, y);
        gradient_norms[iter] = grad_norm;

        fprintf(file, "%d\t%.8f\t%.8f\n", iter + 1, fx_values[iter], gradient_norms[iter]);
        printf("Iteração %d: x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f, passo = %.6f\n", iter + 1, x, y, fx_values[iter], grad_norm, step);

        // Critério de parada
        if ((grad_norm < TOL) || (fabs(dx) < TOL_PONTOS) || (fabs(dy) < TOL_PONTOS) || (delta_f < TOL_FUNC))
        {
            break;
        }
        iter++;
    }

    fclose(file);
    printf("Mínimo encontrado: x = %.6f, y = %.6f, f(x, y) = %.6f\n", x, y, f(x, y));
}

int main()
{
setlocale(LC_ALL, "Portuguese");
setlocale(LC_NUMERIC, "C");

    double x0 = 1, y0 = 1;

    metodo_gradiente_conjugado(funcao, x0, y0, TOL, MAX_ITER, 1, "ConjGrad_SeçãoAurea.txt");
    metodo_gradiente_conjugado(funcao, x0, y0, TOL, MAX_ITER, 2, "ConjGrad_Armijo.txt");

    FILE *gnuplot = popen("gnuplot -persistent", "w");
        fprintf(gnuplot, "set title 'Convergência do Método Gradiente Conjugado'\n");
        fprintf(gnuplot, "set xlabel 'Iteração'\n");
        fprintf(gnuplot, "set ylabel 'Valor'\n");
        fprintf(gnuplot, "set grid\n");
        fprintf(gnuplot, "plot \
            'ConjGrad_SeçãoAurea.txt' using 1:2 title 'f(x,y) - Seção Áurea' with lines, \
            'ConjGrad_SeçãoAurea.txt' using 1:3 title '||grad|| - Seção Áurea' with lines, \
            'ConjGrad_Armijo.txt' using 1:2 title 'f(x,y) - Armijo' with lines, \
            'ConjGrad_Armijo.txt' using 1:3 title '||grad|| - Armijo' with lines\n");
        fflush(gnuplot);
        pclose(gnuplot);

    return 0;
}
