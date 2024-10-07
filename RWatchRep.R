devtools::install_github("zhentaoshi/fsPDA/R_pkg_fsPDA")
library(fsPDA)
data("china_import")
date_import <- names(china_import$treated)

result <- est.fsPDA(
  treated = china_import$treated, 
  control = china_import$control,
  treatment_start = which(date_import == china_import$intervention_time),
  date = as.Date(paste(substr(date_import, 1, 4), "-", 
                       substr(date_import, 5, 6), "-01", sep = ""))
)

plot(result, tlab = "Year", ylab = "Monthly Growth Rate")

result$ATE

# Extract 'treated' and 'control' from china_import
treated <- china_import$treated
control <- china_import$control

# Convert both to dataframes
treated_df <- as.data.frame(treated)
control_df <- as.data.frame(control)

# Concatenate 'treated' and 'control' dataframes
final_df <- cbind(treated_df, control_df)

# Check the resulting dataframe
head(final_df)

# Export the final dataframe as a CSV
write.csv(final_df, "china_import_final.csv", row.names = FALSE)
