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

        System.out.println("Testing DGEMM with offset using JavaBLAS:");
        System.out.println("Matrix A data: " + java.util.Arrays.toString(a));
        System.out.println("Matrix A represents 2x3 starting at offset 1 (elements 4,5,6 and 7,8,9)");
        System.out.println("Matrix B data: " + java.util.Arrays.toString(b));
        System.out.println("Matrix B represents 3x1 (elements 1,2,3)");
        System.out.println("Expected result: [32.0, 50.0] (4*1+5*2+6*3=32, 7*1+8*2+9*3=50)");
        System.out.println();
        
        try {
            // This call previously threw IndexOutOfBoundsException but should now work
            blas.dgemm(
                "N", "N",
                rowsA, colsB, colsA,
                1.0,
                a, offsetA, lda,
                b, offsetB, ldb,
                0.0,
                c, offsetC, ldc
            );
            System.out.println("✓ Multiplication succeeded! Result: " + java.util.Arrays.toString(c));
            
            // Verify the result is correct
            if (Math.abs(c[0] - 32.0) < 1e-10 && Math.abs(c[1] - 50.0) < 1e-10) {
                System.out.println("✓ Result is mathematically correct!");
            } else {
                System.out.println("⚠ Result is incorrect. Got: " + java.util.Arrays.toString(c) + ", expected: [32.0, 50.0]");
            }
        } catch (Exception e) {
            System.out.println("✗ Caught exception: " + e);
            e.printStackTrace();
        }
    }
}