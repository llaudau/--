export void sum1(uniform double x,uniform int i,uniform double result[] ){
    double partial=0.0;
    foreach(j=1 ... 1000000000){
        partial+=1/(j*(j+x));
    }
    result[i]=reduce_add(partial);
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
