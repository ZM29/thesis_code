#This script performs Bai-Perron breakpoint analysis to identify structural breaks.

library(strucchange)
library(readr)

bai_perron_test <- function(data_raw, index_name) {
  data <- ts(data_raw$close)

  #performing the supF and Bai-Perron tests
  supF_test <- sctest(data ~ 1, type = "supF")
  bp_test <- breakpoints(data ~ 1, h = 0.10)
  summary_bp <- summary(bp_test)
  
  #determining the optimal number of breaks and breakpoints based on the BIC
  optimal_breaks <- which.min(summary_bp$RSS[2])
  breakpoint_dates <- breakdates(bp_test)
  
  results <- list(
    breakpoint_dates = breakpoint_dates,
    optimal_breaks = optimal_breaks,
    supF_test = supF_test
  )
  
  return(results)
}

data <- read.csv('datasets/differenced_data.csv')
index_names <- unique(data$Index)

#looping over all the indices
for (index_name in index_names) {
  #filtering and preparing the data
  index_data <- data[data$Index == index_name, ]
  index_data <- index_data[, c("Date", "Close")]
  dates <- index_data$Date
  close <- index_data$Close
  data_raw <- data.frame(close = close, row.names = dates)
  
  #performing the Bai-Perron test
  bp_results <- bai_perron_test(data_raw, index_name)
  
  cat(index_name, "\n")
  print(bp_results$supF_test)
  cat("Optimal number of breaks:", bp_results$optimal_breaks, "\n")
  cat("Breakpoint dates:\n")
  print(bp_results$breakpoint_dates)
}
