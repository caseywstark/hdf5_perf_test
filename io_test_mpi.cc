
#include <cstdlib>
#include <cstddef>
#include <sys/time.h>
#include <iostream>
using namespace std;

#include <mpi.h>
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
Writes the `values` array of size `n` to the file path `output_path`. The IO
is done in chunks of size `n_buffer`.
*/
void write_custom(const int n, const int nx_local,
    const double * const values, const string output_path)
{
    const int nn_local = nx_local * n * n;
    const int num_writers = 32;
    int mpi_rank, mpi_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    // Check that the decomposition is valid.
    if (n % num_writers != 0 && mpi_rank == 0) {
        cerr << "[ERROR] number of writers does not evenly divide the grid size."
            << endl;
        exit(1);
    }

    const int block_size = n * n;
    const int num_blocks = nx_local;
    float *io_buffer = new float[block_size];

    const int num_files = num_writers;
    const int num_rounds = mpi_size / num_files;
    const int file_index = mpi_rank / num_rounds;
    const int ring_position = mpi_rank % num_rounds;
    char local_output_path[1000];
    sprintf(local_output_path, "%s.%04d", output_path.c_str(), file_index);

    // start writing in a ring.
    for (int iring = 0; iring < num_rounds; ++iring) {
        if (iring == ring_position) {
            // Open the file.
            FILE *output_file = fopen(local_output_path, "a");
            if (!output_file) {
                cerr << "[ERROR] could not open file" << output_file << endl;
                exit(1);
            }

            // Fix the position during first write in case the file already exists.
            if (iring == 0) {
                fseek(output_file, 0, SEEK_SET);
            }

            // Write blocks.
            // i0 tracks the current offset in values
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
        MPI_Barrier(comm);
    }
}

/*
Creates HDF5 file `output_path` and writes `values` to a single dataset,
with a rank of `num_dims` and shape `ds_shape`
(that is, `ds_shape` should have `num_dims` elements).
*/
void write_hdf5(const int num_dims, const hsize_t * const ds_shape,
    const int nx_local, const double * const values, const string output_path)
{
    const string dataset_path = "data";
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;

    // Open HDF5 file (trunc mode).
    hid_t pa_plist = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(pa_plist, comm, info);
    hid_t file = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
        pa_plist);
    H5Pclose(pa_plist);

    // check that open was successful.
    if (file > 0) {
        // Create dataspace.
        hid_t grid_dataspace = H5Screate_simple(num_dims, ds_shape, NULL);
        // Create dataset.
        hid_t dataset = H5Dcreate(file, dataset_path.c_str(), H5T_NATIVE_FLOAT,
            dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Grab the dataspace and select the hyperslab
        size_t x0 = mpi_rank * nx_local;
        hsize_t slab_offsets[num_dims] = {x0, 0, 0};
        hsize_t slab_dims[num_dims] = {nx_local, ds_shape[1], ds_shape[2]};
        hid_t status = H5Sselect_hyperslab(grid_dataspace, H5S_SELECT_SET,
            slab_offsets, NULL, slab_dims, NULL);
        if (status < 0) {
            cerr << "[ERROR] could not select hyperslab." << endl;
            exit(1);
        }

        // Write dataset.
        // Create collective IO prop list.
        hid_t write_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(write_plist, H5FD_MPIO_COLLECTIVE);

        // Write the local slab array into the file dataset.
        hid_t slab_dataspace = H5Screate_simple(3, slab_dims, NULL);
        status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, slab_dataspace,
            grid_dataspace, write_plist, data);
        if (status < 0) {
            cerr << "[ERROR] could not write dataset." << endl;
        }

        // Close resources.
        H5Dclose(dataset);
        H5Pclose(write_plist);
        H5Sclose(slab_dataspace);
        H5Sclose(grid_dataspace);
        H5Fclose(file);
    }
    else {
        // we can't write for some reason.
        cerr << "[ERROR] could not open path " << output_path << endl;
    }

    MPI_Barrier(comm);
}

int
main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpi_rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &mpi_rank);

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
    if (mpi_rank == 0) {
        cout << "Running method A." << endl;
    }

    Timer ta;
    ta.start();
    for (int s = 0; s < num_samples; ++s) {
        write_custom(n, data, "test.bin");
    }
    double total_time_a = ta.elapsed();

    // Method B
    if (mpi_rank == 0) {
        cout << "Running method B." << endl;
    }

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
