#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define MEGA_MULTIPLIER    1048576 /* 1024 * 1024 */

static int const block_dim = 3;
static int  rank, nprocs;
static size_t  block_count    = 16;
static size_t  block_size     = 1;
static size_t  block_per_rank = 1;
static size_t  node_count     = 0;
static size_t  core_count     = 0;
static char json[4096]     = "report.json";


void parseArgs (int argc, char * argv[])
{
  char flags[] = "b:k:j:n:c:";
  int opt = 0;
    
  while ((opt = getopt (argc, argv, flags)) != -1) {
    switch ( opt ) {
    case('b'):
      sscanf ( optarg, "%ld", &block_count );
      break;
    case('k'):
      sscanf ( optarg, "%ld", &block_size );
      break;
    case('j'):
      sprintf ( json, "%s", optarg );
      break;
    case('n'):
      sscanf ( optarg, "%ld", &node_count );
      break;
    case('c'):
      sscanf ( optarg, "%ld", &core_count );
      break;
    }
  }
  
  if ( node_count == 0 || core_count == 0 ) {
    if ( rank == 0 )
      fprintf ( stderr, "Node and core counts must be specified!\n" );
    MPI_Abort ( MPI_COMM_WORLD, -1 );
  }
}
  
double * generate ()
{
  size_t i, j;
  double a, b;
  double * restrict array;

  srand ( time(NULL) + rank );
  a = -1000;
  b = 1000;
  
  if ( ! (array = malloc (block_per_rank * block_dim *  sizeof (double))) ) {
    if ( rank == 0 )
      fprintf ( stderr, "malloc failed for %ld blocks per rank!\n", block_per_rank );
    MPI_Abort ( MPI_COMM_WORLD, -1 );
  }
  
  for ( i = 0; i < block_per_rank; i++ )
    for ( j = 0;  j < block_dim;  j++ )
      array [ i * block_dim + j ] =  ( b - a ) * ((double)rand() / RAND_MAX) + a;
  
  return array;
}


void do_shift ( double * restrict array, const double * restrict displacement )
{
  size_t i, j;
  
  for ( i = 0; i < block_per_rank; i++ )
    for ( j = 0;  j < block_dim;  j++ )
      array [ i * block_dim + j ] += displacement [j];
}


void do_average ( double * restrict average, const double * restrict array )
{
  size_t i, j;
  double size;

  size = (double)block_per_rank;  

  for ( j = 0; j < block_dim; j++ )
    average[j] = 0.0;
  
  for ( i = 0; i < block_per_rank; i++ )
    for ( j = 0;  j < block_dim;  j++ )
      average[j] += array [ i * block_dim + j ];

  for ( j = 0; j < block_dim; j++ )
    average[j] /= size;
}


void do_reduce ( double * restrict average )
{
  size_t j;
  double sum_average[block_dim];
  
  MPI_Reduce ( average, sum_average, block_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
  
  if ( rank == 0 )
    for ( j = 0; j < block_dim; j++ )
      average[j] = sum_average[j] / nprocs;
}


int main( int argc, char **argv )
{
  size_t i, j;
  double start_time, end_time, tot_time, max_time;
  double start_generate_time, end_generate_time, tot_generate_time, max_generate_time;
  double start_shift_time, end_shift_time, tot_shift_time, max_shift_time;
  double start_average_time, end_average_time, tot_average_time, max_average_time;
  double start_reduce_time, end_reduce_time, tot_reduce_time, max_reduce_time;
  double * restrict array;
  double displacement[block_dim];
  double average[block_dim];
  MPI_File json_fh;
  MPI_Status status;
  char json_buf[8192];

  MPI_Init( &argc,&argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &nprocs );

  if ( rank == 0 )
    fprintf ( stdout, "Communicator size = %d\n", nprocs);

  parseArgs ( argc, argv );

  block_per_rank = ( block_count * block_size * MEGA_MULTIPLIER ) / nprocs;

  srand ( time(NULL) + rank );
  for ( j = 0;  j < block_dim;  j++ )
    displacement [j] = ((double)rand() / RAND_MAX) + 50;

  start_time = MPI_Wtime();

  start_generate_time = MPI_Wtime();
  array = generate ();
  end_generate_time = MPI_Wtime();

  start_shift_time = MPI_Wtime();
  do_shift ( array, displacement );
  end_shift_time = MPI_Wtime();

  start_average_time = MPI_Wtime();
  do_average ( average, array );
  end_average_time = MPI_Wtime();

  start_reduce_time = MPI_Wtime();
  do_reduce ( average );
  end_reduce_time = MPI_Wtime();

  end_time = MPI_Wtime();

  free(array);

  tot_generate_time = end_generate_time - start_generate_time;
  tot_shift_time    = end_shift_time - start_shift_time;
  tot_average_time  = end_average_time - start_average_time;
  tot_reduce_time  = end_reduce_time - start_reduce_time;
  tot_time          = end_time - start_time;

  MPI_Reduce (&tot_generate_time, &max_generate_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce (&tot_shift_time,    &max_shift_time,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce (&tot_average_time,  &max_average_time,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce (&tot_reduce_time,   &max_reduce_time,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce (&tot_time,          &max_time,          1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if ( rank == 0 ) {
    fprintf ( stdout, "Generate : %.8f s\n", max_generate_time );
    fprintf ( stdout, "Shift    : %.8f s\n", max_shift_time );
    fprintf ( stdout, "Average  : %.8f s\n", max_average_time );
    fprintf ( stdout, "Reduce   : %.8f s\n", max_reduce_time );
    fprintf ( stdout, "Overall  : %.8f s\n", max_time );

    MPI_File_delete ( json, MPI_INFO_NULL );
    MPI_File_open( MPI_COMM_SELF, json, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &json_fh );

    sprintf ( json_buf, "{\n\t\"args\": {\n" );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"block_size\": %ld,\n\t\t\"block\": %ld,\n", block_size, block_count );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"block_per_rank\": %ld,\n", block_per_rank );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"cores\": %ld,\n\t\t\"nodes\": %ld\n\t},\n", core_count, node_count );
    sprintf ( json_buf + strlen (json_buf), "\t\"performance\": {\n" );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"generate\": %.8f,\n", max_generate_time );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"shift\": %.8f,\n", max_shift_time );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"average\": %.8f,\n", max_average_time );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"reduce\": %.8f,\n",  max_reduce_time );
    sprintf ( json_buf + strlen (json_buf), "\t\t\"overall\": %.8f\n\t}\n}\n", max_time );
    
    MPI_File_write ( json_fh, json_buf, strlen(json_buf), MPI_BYTE, &status );
    MPI_File_close ( &json_fh );

    fprintf ( stdout, "\n--> %s\n\n", json );
  }

  MPI_Finalize();

  return 0;
}
  
