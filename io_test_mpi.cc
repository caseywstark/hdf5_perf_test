
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
Writes a 3D grid scalar `values` with shape `(n, n, n)` to `output_path`.
IO is done in chunks of `n * n`.
*/
void write_custom(const int n, const int nx_local,
    const double * const values, const string output_path)
{
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

    // make sure the file is truncated.
    if (ring_position == 0) {
        FILE *output_file = fopen(local_output_path, "w");
        if (!output_file) {
            cerr << "[ERROR] could not open file" << output_file << endl;
            exit(1);
        }
        fclose(output_file);
    }

    // start writing in a ring.
    for (int iring = 0; iring < num_rounds; ++iring) {
        if (iring == ring_position) {
            // Open the file.
            FILE *output_file = fopen(local_output_path, "a");
            if (!output_file) {
                cerr << "[ERROR] could not open file" << output_file << endl;
                exit(1);
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
Writes a 3D grid scalar `values` with shape `(n, n, n)` to `output_path`.
IO is done in chunks of `n * n`.
*/
void write_custom_slow(const int n, const int nx_local,
    const double * const values, const string output_path)
{
    int mpi_rank, mpi_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const int block_size = n * n;
    const int num_blocks = nx_local;
    float *io_buffer = new float[block_size];

    // truncate file.
    if (mpi_rank == 0) {
        FILE *output_file = fopen(output_path.c_str(), "w");
        if (!output_file) {
            cerr << "[ERROR] could not open file" << output_file << endl;
            exit(1);
        }
        fclose(output_file);
    }

    // start writing sequentially.
    for (int irank = 0; irank < mpi_size; ++irank) {
        if (irank == mpi_rank) {
            // Open the file.
            FILE *output_file = fopen(output_path.c_str(), "a");
            if (!output_file) {
                cerr << "[ERROR] could not open file" << output_file << endl;
                exit(1);
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
void write_hdf5(const hsize_t * const ds_shape,
    const int nx_local, const double * const values, const string output_path)
{
    static const int num_dims = 3;
    const string dataset_path = "data";
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

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
            grid_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Grab the dataspace and select the hyperslab
        size_t x0 = mpi_rank * nx_local;
        hsize_t slab_offsets[num_dims] = {0, 0, 0};
        slab_offsets[0] = x0;
        hsize_t slab_dims[num_dims] = {0, ds_shape[1], ds_shape[2]};
        slab_dims[0] = nx_local;
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
            grid_dataspace, write_plist, values);
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
    int mpi_rank, mpi_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    if (argc != 3) {
        cerr << "[ERROR] please provide the grid size and number of timing samples."
            << endl;
    }

    const int n = atoi(argv[1]);
    const int num_samples = atoi(argv[2]);

    // Check that the decomposition is valid.
    if (n % mpi_size != 0 && mpi_rank == 0) {
        cerr << "[ERROR] number of ranks does not evenly divide the grid size."
            << endl;
        exit(1);
    }

    const int nx_local = n / mpi_size;
    const int nn_local = nx_local * n * n;
    double *data = new double[nn_local];
    srand(time(0));
    for (int i = 0; i < nn_local; ++i) {
        data[i] = rand() / double(RAND_MAX);
    }

    // Method A
    if (mpi_rank == 0) {
        cout << "Running method A." << endl;
    }

    Timer ta;
    ta.start();
    for (int s = 0; s < num_samples; ++s) {
        write_custom(n, nx_local, data, "test.bin");
    }
    double total_time_a = ta.elapsed();

    // Slow Method A
    if (mpi_rank == 0) {
        cout << "Running slow method A." << endl;
    }

    ta.start();
    for (int s = 0; s < num_samples; ++s) {
        write_custom_slow(n, nx_local, data, "test.bin");
    }
    double total_time_as = ta.elapsed();

    // Method B
    if (mpi_rank == 0) {
        cout << "Running method B." << endl;
    }

    hsize_t shape[3];
    shape[0] = n; shape[1] = n; shape[2] = n;

    Timer tb;
    tb.start();
    for (int s = 0; s < num_samples; ++s) {
        write_hdf5(shape, nx_local, data, "test.h5");
    }
    double total_time_b = tb.elapsed();

    // Report
    if (mpi_rank == 0) {
        printf("Method A took %f s per call.\n", total_time_a / num_samples);
        printf("Slow Method A took %f s per call.\n", total_time_as / num_samples);
        printf("Method B took %f s per call.\n", total_time_b / num_samples);
    }

    MPI_Finalize();
    return 0;
}
