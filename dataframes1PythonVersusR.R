

existing_cases_file = 'tb_existing_100.csv'

existing_df = read.csv(existing_cases_file,row.names=1,stringsAsFactors=F)

str(existing_df)

existing_df[c(1,2,3,4,5,6,15,16,17,18)]
   
existing_df[c(1,2,3,4,5,6,15,16,17,18)] =
   lapply(existing_df[c(1,2,3,4,5,6,15,16,17,18)],
          function(x) {as.integer(gsub(',','',x))})
# He could have made a function to see if a column had a string,
# and if not, then convert the column ...

str(existing_df)

head(existing_df)

nrow(existing_df)

ncol(existing_df)

# dimensions of data frame are inverted - transpose them

existing_df_t = existing_df

existing_df = as.data.frame(t(existing_df))

head(existing_df,3)

rownames(existing_df)


colnames(existing_df)

# Indexing in R

existing_df[,1]
