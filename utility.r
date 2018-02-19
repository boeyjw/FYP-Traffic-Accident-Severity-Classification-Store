########################################
library(parallel)
cores_2_use <- detectCores() - 3
cl <- makeCluster(cores_2_use)
#########################################
# Impute function
fun_imp <- function(cl, data, X = 1:detectCores() - 2, m = 1, imp_meth, pred_mat, maxit = 5) {
    imp_data <- data[[1]][c(),]
    imp_merge <<- vector(mode = 'list', length = length(data))

    print('Start Imputation')
    for (n in 1:length(data)) {
        imp_pairs <- parLapply(cl = cl, X = X, fun = function(no, d, m, method, predictionMatrix, maxit) {
            mice(d, m = m, printFlag = FALSE, method = method, maxit = maxit, predictorMatrix = predictionMatrix)
        }, data[[n]], m, imp_meth, pred_mat, maxit)

        imp_merge[[n]] <<- imp_pairs[[1]]
        for (p in 2:length(imp_pairs)) {
            imp_merge[[n]] <<- ibind(imp_merge[[n]], imp_pairs[[p]])
        }
        imp_data <- rbind(imp_data, complete(imp_merge[[n]]))

        gc()
        print(n)
    }

    return (imp_data)
}
#########################################
# Split vehicles dataset into multiple years
fun_splt_veh <- function(data, tar) {
    splt_veh <- list()
    tar_names <- names(tar)
    splt_veh[[tar_names[1]]] <- data[data$Accident_Index %in% tar[[1]]$Accident_Index, 2:ncol(data)]
    for (n in 2:length(tar)) {
        splt_veh[[tar_names[n]]] <- data[data$Accident_Index %in% tar[[n]]$Accident_Index, 2:ncol(data)]
    }
    return (splt_veh)
}
#########################################
# Wrapper function to write data to csv file
fun_imp_wrapper <- function(cl, data, file_name, cluster_export_varname, seed = 500, exclude = data.frame(), m = 1, cores_2_use = detectCores() - 2, imp_meth, pred_mat, maxit = 5, rdsexp = NA) {
    if(nrow(exclude) > 0 && (nrow(exclude) != sum(sapply(data, nrow)))) {
        print('Unbindable data & exclusion rows')
        stopCluster(cl)
        return
    }
    library(mice)
    library(dplyr)

    clusterSetRNGStream(cl, seed)
    clusterExport(cl, cluster_export_varname)
    clusterEvalQ(cl, library(mice))

    imp_data <- fun_imp(cl = cl, data = data, X = 1:cores_2_use, m = m, imp_meth = imp_meth, pred_mat = pred_mat, maxit = maxit)
    stopCluster(cl)

    if(nrow(exclude) > 0)
        imp_data_full <- bind_cols(imp_data, exclude)

    write.csv(x = imp_data_full, file = paste0(file_name, '.imp.csv'), row.names = FALSE)
    if(!is.na(rdsexp))
        saveRDS(imp_merge, paste0(rdsexp, '.RData'))

}
########################################
