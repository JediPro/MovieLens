# Set environs ####
unzip('ml-10m.zip')
library(data.table)
library(car)
library(tm)
library(Matrix)
library(slam)
library(recommenderlab)

# check structure of files
readLines('ratings.dat', 10)
readLines('movies.dat', 10)
readLines('users.dat', 10)

# read in movies files
movies_raw = readLines('movies.dat')
movies_raw = gsub(pattern = "::", replacement = ";", movies_raw)
writeLines(movies_raw, "movies.csv")
closeAllConnections()
movies = fread('movies.csv', sep = ";", header = F)
rm(movies_raw)
colnames(movies) = c('MovieID', 'Title', 'Genres')

# read in ratings file
ratings_raw = readLines('ratings.dat')
ratings_raw = gsub("::", ";", ratings_raw)
writeLines(ratings_raw, 'ratings.csv')
closeAllConnections()
ratings = fread('ratings.csv', sep = ";", header = F)
rm(ratings_raw)
colnames(ratings) = c('UserID', 'MovieID', 'Rating', 'Timestamp')

# read in tags data
users_raw = readLines('users.dat')
users_raw = gsub("::", ";", users_raw)
writeLines(users_raw, 'users.csv')
users = read.csv('users.csv', sep = ";", header = F)
rm(users_raw)
colnames(users) = c('UserID', 'Gender', 'Age', 'Occupation', 'Zip')

# Pre process data ####
# Ratings convert timestamp to Date
ratings[, DateTime := as.POSIXct(Timestamp, origin = '1970-01-01')]
ratings[, Date := as.Date(DateTime)]
# Movies extract year from title
movies[, Year := substr(Title, start = nchar(Title) - 4, nchar(Title) - 1)]
movies[, Year := as.integer(Year)]
# Movies form term frequency matrix for genres
# Create term document matrix
GenreTDM = TermDocumentMatrix(Corpus(VectorSource(movies$Genres)))
# Overall counts of eeach genre to filter out rarest ones
GenreFreq = data.frame(Term = rownames(GenreTDM), Freq = row_sums(GenreTDM))
# remove terms which are rare
GenreTDM = GenreTDM[!rownames(GenreTDM) %in% c('film'),]
# convert TDM to DTM to add back to dataset
GenreDTM = t(GenreTDM)
GenreDTM = as.matrix(GenreDTM)
colnames(GenreDTM)
GenreDTM = as.data.frame(GenreDTM)
movies = cbind(movies, GenreDTM)
rm(GenreDTM, GenreTDM)

# remove unwanted columns froom ratings
ratings[,DateTime := NULL]
ratings[,Timestamp := NULL]

# Content Base Recommendation ####

# find movies that have not been rated
setdiff(movies$MovieID, ratings$MovieID)
# remove movies from movies which have not been rated
movies = movies[!(MovieID %in% setdiff(movies$MovieID, ratings$MovieID)) ]

# Create dummy variables in movies for year of release
movies$Decade = recode(as.numeric(movies$Year), "1900:1939 = 'd30'; 1940:1949 = 'd40'; 1950:1959 = 'd50'; 
1960:1969 = 'd60';1970:1979 = 'd70'; 1980:1989 = 'd80'; 1990:1999 = 'd90'; 2000:2009 = 'd00'")
# create movie ID wise dummy variables for decades
DecadeDTM = DocumentTermMatrix(Corpus(VectorSource(movies$Decade)))
DecadeDTM = as.data.frame(as.matrix(DecadeDTM))
rm(DecadeDTM, GenreFreq)
# add back to movies data table
movies = cbind(movies, DecadeDTM)
movies = as.data.table(movies)
# remove Decades column
movies[, Decade := NULL]

# multiply ratings and movie matrices to get user preference vectors for features
MovieVectors = movies[,c(5:30)]
MovieVectors = t(MovieVectors)

# convert matrix to sparse
movies_sparse = as(MovieVectors, "sparseMatrix")

# convert ratings colmns to factors to use in conversion to sparse matrix
ratings[, ':=' (UserID = as.factor(UserID), MovieID = as.factor(MovieID))]
# convert long form of ratings directly to wide
ratings_sparse = with(ratings[,-4], sparseMatrix(i = as.numeric(MovieID), j = as.numeric(UserID), 
                                                 x = as.numeric(Rating), 
                                                 dimnames = list(levels(MovieID), levels(UserID))))

# Compute dot product of ratings and features per user
UserVectors_sparse = MovieVectors %*% ratings_sparse
UserVectors = as.data.table(as.matrix(UserVectors_sparse))
rm(movies_sparse, ratings_sparse, UserVectors_sparse)

# Convert each row to 0-1 scale
UserVectors_scaled = scale(UserVectors)
UserVectors_scaled = as.data.table(UserVectors_scaled)

# Multiply user and movie vectors to get a user-movie matrix
UserVectors_scaled = as(as.matrix(UserVectors_scaled), 'sparseMatrix')
UserMovieVectors = t(movies_sparse) %*% UserVectors_scaled
UserMovieVectors = as.data.table(as.matrix(UserMovieVectors))
rm(UserVectors_scaled, UserVectors_sparse, ratings_sparse, MovieVectors, movies_sparse, movies3)
rm(UserVectors)

# Collaborative Filtering ####
# convert ratings colmns to factors to use in conversion to sparse matrix
ratings[, ':=' (UserID = as.factor(UserID), MovieID = as.factor(MovieID))]
# convert long form of ratings directly to wide
ratings_sparse = with(ratings[,-4], sparseMatrix(i = as.numeric(MovieID), j = as.numeric(UserID), 
                                                 x = as.numeric(Rating), 
                                                 dimnames = list(levels(MovieID), levels(UserID))))
# convert to Real Rating Matrix class
ratings_sparse_real = as(ratings_sparse, "realRatingMatrix")
rm(ratings_sparse)
rm(UserVectors)

# Normalize data
ratings_norm = normalize(ratings_sparse_real)

# Create Recommender Model
recom_model = Recommender(data = ratings_norm, method = 'UBCF', param = list(method = 'Cosine', nn = 30))
recom_model_pred = predict(recom_model, ratings_norm[1], n = 10)
recom_list = as(recom_model_pred, 'list')

# See recommendations
movies$Title[as.integer(recom_list[[1]][1])]

# Evaluate model
recom_eval = evaluationScheme(ratings_sparse_real, method = 'cross-validation', k = 5, given = 1, 
                              goodRating = 5)
eval_results = evaluate(recom_eval, method = 'UBCF', n = c(1, 5, 10))
eval_conf = getConfusionMatrix(eval_results)
