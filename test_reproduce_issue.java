import dev.ludovic.netlib.blas.JavaBLAS;
import dev.ludovic.netlib.blas.BLAS;

public class test_reproduce_issue {
    public static void main(String[] args) {
        // Raw data for matrix A (2 rows, 3 cols), offset = 1, data length = 9
        BLAS blas = JavaBLAS.getInstance();

        double[] a = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
        int rowsA = 2;
        int colsA = 3;
        int lda = 3; // Leading dimension (colStride)
        int offsetA = 1;

        // Raw data for matrix B (3 rows, 1 col), offset = 0, data length = 3
        double[] b = {1.0, 2.0, 3.0};
        int rowsB = 3;
        int colsB = 1;
        int ldb = 3; // Leading dimension (colStride)
        int offsetB = 0;

        // Output matrix C (2 rows, 1 col)
        double[] c = new double[2];
        int rowsC = 2;
        int colsC = 1;
        int ldc = 2; // Leading dimension for C
        int offsetC = 0;

        try {
            // This call should reproduce the IndexOutOfBoundsException if the offset logic is incorrect
            blas.dgemm(
                "N", "N",
                rowsA, colsB, colsA,
                1.0,
                a, offsetA, lda,
                b, offsetB, ldb,
                0.0,
                c, offsetC, ldc
            );
            System.out.println("Multiplication succeeded, result:");
            for (double v : c) {
                System.out.print(v + " ");
            }
            System.out.println();
        } catch (Exception e) {
            System.out.println("Caught exception: " + e);
            e.printStackTrace();
        }
    }
}