#include <stdio.h>
#include <math.h>
#include <locale.h>

#define MAX_ITER 1000
#define TOL 1e-5

double funcao(double x, double y) {
    return 5*pow(x, 2) + 5*pow(y, 2) - x*y - 11*x + 11*y + 11;
}

void gradiente(double (*f)(double, double), double x, double y, double grad[]) {
    double h = 1e-6;

    grad[0] = (f(x + h, y) - f(x, y)) / h; // partial x
    grad[1] = (f(x, y + h) - f(x, y)) / h; // partial y
}

double armijo(double (*f)(double, double), double x, double y, double grad[], double alpha, double beta) {
    double f_curr = f(x, y);
    double f_new;

    while (1)
        {
        double new_x = x - alpha * grad[0];
        double new_y = y - alpha * grad[1];
        f_new = f(new_x, new_y);

        if (f_new <= f_curr - beta * alpha * (grad[0] * grad[0] + grad[1] * grad[1]))
        {
            break;
        }
        alpha *= 0.5;
    }

    return alpha;
}

void metodoGradiente(double (*f)(double, double), double x, double y, double tol, double delta_tol, double f_tol, int max_iter, double alpha, double beta) {
    double grad[2];
    double fx_values[max_iter];
    double gradient_norms[max_iter];
    int iter = 0;

    gradiente(f, x, y, grad);
    double grad_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);

    fx_values[0] = f(x, y);   // iteração 0
    gradient_norms[0] = grad_norm;

    printf("Iteração 0 (início da busca): x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f\n", x, y, fx_values[0], gradient_norms[0]);

    while (iter < max_iter) {
        double prev_x = x;
        double prev_y = y;
        double prev_f = f(x, y);

        gradiente(f, x, y, grad);
        grad_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1]);
        if (grad_norm < tol) break;

        fx_values[iter] = f(x, y);
        gradient_norms[iter] = grad_norm;

        alpha = armijo(f, x, y, grad, alpha, beta);

        x -= alpha * grad[0];
        y -= alpha * grad[1];
        double curr_f = f(x, y);

        // critério de parada
        if (fabs(x - prev_x) < delta_tol || fabs(y - prev_y) < delta_tol || fabs(curr_f - prev_f) < f_tol) break;

        printf("Iteração %d: x = %.6f, y = %.6f, f(x, y) = %.6f, ||grad|| = %.6f, alpha = %.6f\n", iter + 1, x, y, curr_f, grad_norm, alpha);

        iter++;
    }

    printf("Mínimo encontrado pelo método do gradiente em x = %.6f, y = %.6f, f(x, y) = %.6f\n", x, y, f(x, y));

    // análise de convergência
    FILE *file = fopen("INf3_Grad_Ar.txt", "w");
    if (file) {
        fprintf(file, "# Iteração\tf(x, y)\t||gradiente||\n");
        for (int i = 0; i <= iter; i++) {
            fprintf(file, "%d\t%.8f\t%.8f\n", i + 1, fx_values[i], gradient_norms[i]);
        }
        fclose(file);
    }

    // plot da análise de convergência
    FILE *gnuplot_cv = popen("gnuplot -persistent", "w");
    if (gnuplot_cv) {
        fprintf(gnuplot_cv, "set title 'Análise de Convergência do Método do Gradiente'\n");
        fprintf(gnuplot_cv, "set xlabel 'Iteração'\n");
        fprintf(gnuplot_cv, "set ylabel 'Valor'\n");
        fprintf(gnuplot_cv, "set key top right\n");
        fprintf(gnuplot_cv, "set grid\n");

        double f_minimo = f(x, y);

        fprintf(gnuplot_cv, "plot 'INf3_Grad_Ar.txt' using 1:2 title 'f(x, y)' with lines lw 2, '' using 1:3 title '||gradiente||' with lines lw 2, ");
        fprintf(gnuplot_cv, "0 with lines lt 2 lc rgb 'red' title 'y = 0', ");
        fprintf(gnuplot_cv, "%f with lines lt 2 lc rgb 'black' title 'y = mínimo'\n", f_minimo);

        fflush(gnuplot_cv);
        pclose(gnuplot_cv);
    }
}

int main() {
    setlocale(LC_ALL, "Portuguese");
    setlocale(LC_NUMERIC, "C");

    double x = 1.0;      // condições iniciais
    double y = 1.0;
    double alpha = 0.1;  // taxa de aprendizado inicial
    double beta = 0.1;   // parâmetro Armijo
    double tol = 1e-6;   // tolerâncias
    double delta_tol = 1e-6;
    double f_tol = 1e-6;
    int max_iter = 1000;

    metodoGradiente(funcao, x, y, tol, delta_tol, f_tol, max_iter, alpha, beta);

    return 0;
}
