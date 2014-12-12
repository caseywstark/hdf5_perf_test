
#include <cstdlib>
#include <cstddef>
#include <sys/time.h>
#include <iostream>
using namespace std;

#include <hdf5.h>

typedef struct timeval tv;

class Timer {
  public:
    tv t0;
    Timer() { }
    void start() {
        gettimeofday(&t0, NULL);
    }
    double elapsed() {
        tv t1;
        gettimeofday(&t1, NULL);
        double dt = (t1.tv_sec - t0.tv_sec) + 1.0e-6 * (t1.tv_usec - t0.tv_usec);
        return dt;
    }
};

/*
Writes a 3D grid scalar `values` with shape `(n, n, n)` to `output_path`.
IO is done in chunks of `n * n`.
*/
void write_custom(const int n, const double * const values,
    const string output_path)
{
    const int block_size = n * n;
    const int num_blocks = n;
    float *io_buffer = new float[block_size];

    // use C-style IO.
    FILE *output_file = fopen(output_path.c_str(), "w");
    // check that open was successful.
    if (output_file) {
        // i0 tracks the current offset in values.
        int i0 = 0;
        for (int iblock = 0; iblock < num_blocks; ++iblock) {
            // Copy to I/O buffer.
            for (int i = 0; i < block_size; ++i) {
                io_buffer[i] = values[i0 + i];
            }
            i0 += block_size;
            fwrite(io_buffer, sizeof(float), block_size, output_file);
        }
        // Close file.
        fclose(output_file);
    }
    else {
        // we can't write for some reason.
        cerr << "[ERROR] could not open path " << output_path << endl;
    }
}

/*
Creates HDF5 file `output_path` and writes `values` to a single dataset,
with a rank of `num_dims` and shape `ds_shape`
(that is, `ds_shape` should have `num_dims` elements).
*/
void write_hdf5(const int num_dims, const hsize_t * const ds_shape,
    const double * const values, const string output_path)
{
    const string dataset_path = "data";

    // Open HDF5 file (trunc mode).
    hid_t file = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
        H5P_DEFAULT);
    // check that open was successful.
    if (file > 0) {
        // Create dataspace.
        hid_t dataspace = H5Screate_simple(num_dims, ds_shape, NULL);
        // Create dataset.
        hid_t dataset = H5Dcreate(file, dataset_path.c_str(), H5T_NATIVE_FLOAT,
            dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        // Write dataset.
        hid_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, dataspace,
            dataspace, H5P_DEFAULT, values);
        if (status < 0) {
            cerr << "[ERROR] could not write dataset." << endl;
        }

        // Close resources.
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Fclose(file);
    }
    else {
        // we can't write for some reason.
        cerr << "[ERROR] could not open path " << output_path << endl;
    }
}

int
main(int argc, char **argv)
{
    if (argc != 3) {
        cerr << "[ERROR] please provide the grid size and number of timing samples."
            << endl;
    }

    const int n = atoi(argv[1]);
    const int num_samples = atoi(argv[2]);

    const int nn = n * n * n;
    double *data = new double[nn];
    srand(time(0));
    for (int i = 0; i < nn; ++i) {
        data[i] = rand() / double(RAND_MAX);
    }

    // Method A
    cout << "Running method A." << endl;

    Timer ta;
    ta.start();
    for (int s = 0; s < num_samples; ++s) {
        write_custom(n, data, "test.bin");
    }
    double total_time_a = ta.elapsed();

    // Method B
    cout << "Running method B." << endl;

    int num_dims = 3;
    hsize_t shape[3];
    shape[0] = n; shape[1] = n; shape[2] = n;

    Timer tb;
    tb.start();
    for (int s = 0; s < num_samples; ++s) {
        write_hdf5(num_dims, shape, data, "test.h5");
    }
    double total_time_b = tb.elapsed();

    // Report
    printf("Method A took %f s per call.\n", total_time_a / num_samples);
    printf("Method B took %f s per call.\n", total_time_b / num_samples);

    return 0;
}
