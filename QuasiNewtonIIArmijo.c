#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-6
#define TOL_PONTOS 1e-6
#define TOL_FUNC 1e-6

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

// Regra de Armijo
double armijo(double (*f)(double, double), double x, double y, double direcao[], double alpha, double beta)
{
    double f_curr = f(x, y);
    double f_new;
    while (1)
    {
        double new_x = x + alpha * direcao[0];
        double new_y = y + alpha * direcao[1];
        f_new = f(new_x, new_y);

        if (f_new <= f_curr - beta * alpha * (direcao[0] * direcao[0] + direcao[1] * direcao[1])) break;
        alpha *= 0.5;
    }
    return alpha;
}

// Quasi-Newton BFGS
void metodoBFGS(double (*f)(double, double), double x0, double y0, double tol, int max_iter, double *f_minimo)
{
    double x = x0, y = y0;
    double grad_x, grad_y, grad_x_ant, grad_y_ant;
    double H[2][2] = {{1, 0}, {0, 1}}; // Matriz Identidade
    double fx_values[MAX_ITER], gradient_norms[MAX_ITER];
    double last_f_value = f(x, y);

    gradiente(f, x, y, &grad_x, &grad_y);

    FILE *file = fopen("f5_NewtonBFGS_Ar.txt", "w");
    fprintf(file, "# Iteração\tf(x)\t||gradiente||\n");
    fprintf(file, "0\t%.8f\t%.8f\n", f(x, y), sqrt(grad_x * grad_x + grad_y * grad_y));

    for (int iter = 0; iter < max_iter; iter++)
    {
        grad_x_ant = grad_x;
        grad_y_ant = grad_y;

        double d1 = -(H[0][0] * grad_x + H[0][1] * grad_y);
        double d2 = -(H[1][0] * grad_x + H[1][1] * grad_y);
        double direcao[2] = {d1, d2};
        double alpha = armijo(f, x, y, direcao, 1.0, 0.1);

        double x_ant = x;
        double y_ant = y;
        x += alpha * d1;
        y += alpha * d2;

        gradiente(f, x, y, &grad_x, &grad_y);

        double s1 = x - x_ant;
        double s2 = y - y_ant;
        double y1 = grad_x - grad_x_ant;
        double y2 = grad_y - grad_y_ant;

        double ys = y1 * s1 + y2 * s2;

        if (fabs(ys) < 1e-8)
        {
            H[0][0] = 1; H[0][1] = 0; H[1][0] = 0; H[1][1] = 1;
            printf("Recondicionamento de H na iteração %d.\n", iter); // Recondicionamento utiliza matriz identidade
            continue;
        }

        double rho = 1.0 / ys;
        double Hy1 = H[0][0] * y1 + H[0][1] * y2;
        double Hy2 = H[1][0] * y1 + H[1][1] * y2;
        H[0][0] += rho * (s1 * s1) - rho * (Hy1 * s1);
        H[0][1] += rho * (s1 * s2) - rho * (Hy1 * s2);
        H[1][0] += rho * (s2 * s1) - rho * (Hy2 * s1);
        H[1][1] += rho * (s2 * s2) - rho * (Hy2 * s2);

        fx_values[iter] = f(x, y);
        gradient_norms[iter] = sqrt(grad_x * grad_x + grad_y * grad_y);

        printf("Iteração %d: (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", iter + 1, x, y, f(x, y));

        double dx = fabs(x - x_ant);
        double dy = fabs(y - y_ant);
        double delta_f = fabs(f(x, y) - f(x_ant, y_ant));
        double grad_norm = sqrt(grad_x * grad_x + grad_y * grad_y);

        // Relaxamento no critério de parada
        double progress_ratio = (last_f_value - f(x, y)) / last_f_value;
        last_f_value = f(x, y);

        // Critério de parada - permite mais iterações se ainda houver variação na função objetivo
        if ((grad_norm < tol) || (dx < tol) || (dy < tol) || (delta_f < tol) || progress_ratio < 1e-4)
        {
            printf("Critério de parada alcançado na iteração %d.\n", iter + 1);
            break;
        }

        fprintf(file, "%d\t%.8f\t%.8f\n", iter + 1, fx_values[iter], gradient_norms[iter]);
    }

    fclose(file);

    *f_minimo = f(x, y);
    printf("Mínimo encontrado em (x, y) = (%.6f, %.6f), f(x, y) = %.6f\n", x, y, *f_minimo);

}

int main()
{
setlocale(LC_ALL, "Portuguese");
    double x0 = 1.0, y0 = 1.0;
    double f_minimo;

    metodoBFGS(funcao, x0, y0, TOL, MAX_ITER, &f_minimo);

    FILE *gnuplot_cv = popen("gnuplot -persistent", "w");
        fprintf(gnuplot_cv, "set title 'Análise de Convergência do Método de Quasi-Newton com Armijo'\n");
        fprintf(gnuplot_cv, "set xlabel 'Iteração'\n");
        fprintf(gnuplot_cv, "set ylabel 'Valor'\n");
        fprintf(gnuplot_cv, "set key top right\n");
        fprintf(gnuplot_cv, "set grid\n");
        fprintf(gnuplot_cv, "plot 'f5_NewtonBFGS_Ar.txt' using 1:2 with lines title 'f(x, y)', '' using 1:3 with lines title '||gradiente||'\n");

        fflush(gnuplot_cv);
        pclose(gnuplot_cv);

    return 0;
}
