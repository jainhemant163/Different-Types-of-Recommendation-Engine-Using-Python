Popularity Based Recommender
---------------------------------------------------------------

#importing libraries
import pandas as pd
import numpy as np

#reading the files
data = pd.read_csv('listing.csv', encoding = 'latin-1')
books = pd.read_csv('books.csv', encoding = 'latin-1')

#using head() function to view first 5 rows for the object based on position. 
Just to test if we have right data.
data.head()

books.head()

# Getting recommendation based on No. Of ratings
rating_count = pd.DataFrame(books, columns=['book_id','no_of_ratings'])
# Sorting and dropping the duplicates
rating_count.sort_values('no_of_ratings', ascending=False).drop_duplicates().head(10)

# getting the detail of 5 most rated books
most_rated_books = pd.DataFrame([4755, 2409, 2194, 4696, 1616], index=np.arange(5), columns=['book_id'])
detail = pd.merge(most_rated_books, data, on='book_id')
detail

# getting the most rated book
most_rated_book = pd.DataFrame(books, columns=['book_id', 'user_id', 'avg_rating', 'no_of_ratings'])
most_rated_book.max()

#getting description for most rated book
most_rated_book.describe()

# description for author
data['author'].describe()


------------------------------------------------
Correlation Based Recommender
------------------------------------------------
# importing libraries
import pandas as pd
import numpy as np

# reading files
data = pd.read_csv('listing.csv', encoding = 'latin-1')
books = pd.read_csv('books.csv', encoding = 'latin-1')

# Checking the data using head function
books.head()

# calculating the mean
rating = pd.DataFrame(books.groupby('book_id')['no_of_ratings'].mean())
rating.head()

# getting the description of rating
rating.describe()

# sorting based on no of ratings that each book got
rating.sort_values('no_of_ratings', ascending=False).head()

# Preparing data table for analysis
ratings_pivot = pd.pivot_table(data=books, values='user_rating', index='user_id', columns='book_id')
ratings_pivot.head()


#As we are interested in finding correlation between two variables, for that, we are going to use Pearson correlation 
#which would simply measure the linear correlation. In this case, we are interested in knowing the relation between 
#two books based on user rating.

correlation_matrix  = user_rating.corr(method='pearson')
correlation_matrix.head(10)

# getting the users who rated this particular book (most rated) and making sure rating is not zero
OneManOut_rating = ratings_pivot[4755]
OneManOut_rating[OneManOut_rating>=0]


# finidng similar books to One Man Out book using Pearson correlation
similar_to_OneManOut = ratings_pivot.corrwith(OneManOut_rating)
corr_OneManOut = pd.DataFrame(similar_to_OneManOut, columns=['PearsonR'])
corr_OneManOut.dropna(inplace=True)
corr_OneManOut.head()

OneManOut_corr_summary = corr_OneManOut.join(rating)

# getting the most similar book
OneManOut_corr_summary.sort_values('PearsonR', ascending=False).head(10)

# getting the details for most similar books
book_corr_OneManOut = pd.DataFrame([2629, 493, 4755, 4571, 2900, 1417, 2681, 1676, 2913, 1431], 
                      index = np.arange(10), columns=['book_id'])
summary = pd.merge(book_corr_OneManOut, data,on='book_id')
summary


-------------------------------------
Content Base Recommender
-------------------------------------


#There exists another type of recommender known as content based recommender. This type of recommender uses the description of the item to recommend next most similar item. Content based recommenders also make the ‘personalized’ recommendation. The main difference between correlation based recommender and content based recommender is that the former considers the ‘user behavior’ while later considers the content for making recommendation. Content based recommender uses the product features or keywords used in description to find the similarity between the items. Let’s see how can we build one.


# importing libraries
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectori

#linear_kernel is used to compute the linear kernel between two variables. We would use this function instead of 
#cosine_similarities() because it is faster and as we are also using TF-IDF vectorization, a simple dot product 
#will give us the same cosine similarity score. Now what is TF-IDF vector? We cannot compute the similarity 
#between the given description in the form it is in our dataset. This is practically impossible. 
#For this purpose, Term Frequency-Inverse Document Frequency (TF-IDF) is calculated for all the documents 
#which would simply return you a matrix with each word representing a column. sklearn’s TfidfVectorizer would 
#do this for us in a couple of lines:

# reading file
book_description = pd.read_csv('description.csv', encoding = 'latin-1')

# checking if we have the right data
book_description.head()

# removing the stop words
books_tfidf = TfidfVectorizer(stop_words='english')
# filling the missing values with empty string
book_description['description'] = book_description['description'].fillna('')
# computing TF-IDF matrix required for calculating cosine similarity
book_description_matrix = books_tfidf.fit_transform(book_description['description'])


# Let's check the shape of computed matrix
book_description_matrix.shape

#The above shape means that 4186 words are used to describe 143 books in our dataset. 

# computing cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(book_description_matrix, book_description_matrix)

indices = pd.Series(book_description['name'].index)

# Function to get the most similar books
def recommend(index, cosine_sim=cosine_similarity):
    id = indices[index]
    # Get the pairwsie similarity scores of all books compared to that book, 
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_sim[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]

    # Get the books index
    books_index = [i[0] for i in similarity_scores]

    # Return the top 5 most similar books using integer-location based indexing (iloc)
    return book_description['name'].iloc[books_index]
	

# getting recommendation for book at index 2
recommend(2)

# getting recommendation for book at index 6
recommend(6)





