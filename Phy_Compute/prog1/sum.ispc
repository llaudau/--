export void sum1(uniform double x,uniform int i,uniform double result[] ){
    double partial=0.0;
    double c=0.0;
    foreach(j=1 ... 100000000){
        double term=1/(j*(j+x));
        double y=term-c;
        double t=partial+y;
        c+=(t-partial)-y;
        partial=t;



    }
    result[i]=reduce_add(partial)+reduce_add(c);
};

// extern "C" {
//     uniform double sum1(uniform double x) {
//         uniform double result;
//         double partial = 0.0;
//         foreach(j = 1 ... 100000) {
//             partial += 1.0 / (j * (j + x));
//         }
//         result = reduce_add(partial);  
//         return result;
//     }
// }


// extern "C" {
//     uniform double sum1(uniform double x) {
//         uniform double result = 0.0;
//         double partial = 0.0;

//         foreach (j = 1 ... 100000) {
//             partial += 1.0 / (j * (j + x));
//         }

//         result = reduce_add(partial);  // Reduce the partial results across all threads
//         return result;
//     }
// }
