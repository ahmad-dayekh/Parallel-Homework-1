#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to evaluate the curve (y = f(x))
float f(float x, long long int* comp_step_count) {
    (*comp_step_count)++; // counting multiplication in x * x
    return x * x; // Example: y = x^2
}

// Function to compute the area of a trapezoid
float trapezoid_area(float a, float b, float d, long long int* comp_step_count) { 
    float area = 0;
    for (float x = a; x < b; x += d) {
        float x_d = x + d;
        (*comp_step_count)++; // counting addition in x + d

        float fx = f(x, comp_step_count);

        float fx_d = f(x_d, comp_step_count);

        (*comp_step_count)++; // counting addition in fx + fx_d
        float sum = fx + fx_d;

        (*comp_step_count)++; // counting addition in area += sum
        area += sum;

        (*comp_step_count)++; // counting addition in x += d (in the loop)
    }
    (*comp_step_count)++; // counting multiplication in area * d
    (*comp_step_count)++; // counting division in (area * d) / 2.0f
    return area * d / 2.0f;
}

int main(int argc, char** argv) {
    int rank, size;
    float a = 0.0f, b = 1.0f;  // Limits of integration
    int n;
    float start, end, local_area, total_area;
    long long int local_comp_step_count = 0;
    long long int total_comp_step_count = 0;

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (rank == 0) {
        // Get the number of intervals from the user
        printf("Enter the number of intervals: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    // Broadcast the number of intervals to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float b_minus_a = b - a;
    local_comp_step_count++; // counting subtraction in b - a

    float d = b_minus_a / n;
    local_comp_step_count++; // counting division in (b - a) / n

    float region = b_minus_a / size;
    local_comp_step_count++; // counting division in (b - a) / size

    float rank_region = rank * region;
    local_comp_step_count++; // counting multiplication in rank * region

    start = a + rank_region;
    local_comp_step_count++; // counting addition in a + rank_region

    end = start + region;
    local_comp_step_count++; // counting addition in start + region

    // Each process calculates the area of its subinterval
    local_area = trapezoid_area(start, end, d, &local_comp_step_count);

    // Reduce all local areas to the total area on the root process
    MPI_Reduce(&local_area, &total_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reduce the comp_step counts to the total comp_step count
    MPI_Reduce(&local_comp_step_count, &total_comp_step_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The total area under the curve is: %f\n", total_area);
        printf("Total computation steps: %lld\n", total_comp_step_count);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}