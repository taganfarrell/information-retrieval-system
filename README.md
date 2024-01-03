# information-retrieval-system
An efficient and comprehensive Information Retrieval System capable of creating diverse indexes, implementing three retrieval models (Cosine Similarity, BM25, and Language Model), and incorporating advanced features such as Pseudo Relevance Feedback and Query Expansion for improved document retrieval.

Language: python (3.9.6)

Only standard library functions used

How to run:
To build each index:

mainP2.py build [trec-files-directory-path] [index-type] [output-dir]

index type options are: ‘single’, ‘phrase’, ‘positional’, ‘stem’, and ‘doc_term’

Example:
python3 /Users/taganfarrell/PycharmProjects/pythonProject/mainP2.py build /Users/taganfarrell/PycharmProjects/pythonProject/BigSample/ doc_term /Users/taganfarrell/PycharmProjects/pythonProject/Output/

To static query:
mainP2.py query_static [index-directory-path] [query-file-path] [retrieval-model] [index-type] [results-file] [ranking-type]

index type options are: ‘single’ and ‘stem’
retrieval model options are: ‘cosine’, ‘bm25’, and ‘lm’
ranking-type options are: ‘regular’, ‘prf’, ‘reduction’, and ‘prf_reduction’

The ‘regular’ ranking type is used to run rankings normally with each model. The other ranking type options are to perform pseudo relevance feedback, query reduction, and PRF and reduction, respectively.

Example:
python3 /Users/taganfarrell/PycharmProjects/pythonProject/mainP2.py query_static /Users/taganfarrell/PycharmProjects/pythonProject/Output/ /Users/taganfarrell/PycharmProjects/pythonProject/queryfile.txt bm25 single /Users/taganfarrell/PycharmProjects/pythonProject/Output/BM25SingleReduction.txt

The basics of the program works largely the same as before. I added code to create a  ‘doc_term” index and an idf dictionary that are both written to the output folder. These are used in the PRF calculations to gather the top terms from the top documents of each query.

I added a command line argument and if else blocks within my main function to allow the choice of different ranking systems when performing static queries. To perform all of the regular rankings with Cosine, BM25, and LM as before, you simply add ‘regular’ as the last command line argument. To perform pseudo relevance feedback you use ‘prf.’ To perform query reduction you use ‘reduction.’ And to perform both of them you use ‘prf_reduction.’

When pseudo relevance feedback is called, it first performs bm25 ranking on the queries just as before. Then, I retrieve the top 20 documents for each query with the function, retrieve_top_20_docs(results_file). I then pass those 20 documents to another function that calculates the good terms from that set of documents using both num.idf and num.ntf.idf. I also pass a list of N-values and a list of M-values to that function. These lists of values for each parameter can be adjusted in main but are defaulted to all the listed necessary values from the project document. With nested for loops, it calculates the top terms for every combination of N documents and M terms and for each criteria of num.idf and num.ntf.idf and places them in a dictionary. Afterwards, the function expand_and_rescore_queries takes the original queries dictionary, adds the top terms and their tf-idf values to each query, and then recalculates the bm25 score for each query against every document. This is again done with nested for loops for every combination of parameters and the expanded query results are returned. Lastly, the results are sent to another function that for each query and each set of parameters ranks the documents and writes the top 100 documents to an output file with the associated N, M, and criteria values in the output directory.
Example output: results_N=10_M=5_num_ntf.idf.txt

For query reduction, I created a new function to process the queries. This method of ranking looked at the narrative text of the query file instead of the title. The new function worked largely the same as the original function to read the query file except for taking the <narr> instead of <title> and also filtered out terms based on a threshold percentage. Once the text had been tokenized and tf-idf values were calculated, the top X percentage of tf-idf values were kept and placed into the reduced query dictionary. I tried a few different threshold values and the one that appeared to produce the highest MAP score was 35%. Once this reduced query dictionary was made, it was simply passed to the respective functions to compute and rank and write bm25 scores in the same way that the original bm25 was programmed. I also tried a few other methods for the threshold. I have commented out lines of code that include keeping terms over a certain value for tf-idf, such as only keeping those above a value of 3. I also thought about just keeping the top X terms, such as simply keeping the top 5 highest tf-idf values. However, I decided to use the percentage threshold in my final code.

For PRF and reduction on long queries, the program essentially just runs both sections of code one after the other. It does query reduction in the exact same way as above and ranks the documents with the reduced queries. Pseudo relevance feedback is then done on those results to add potentially relevant terms back into the reduced queries and ranks the new queries with bm25 again.

Report and Analysis 

Retrieval Model:
BM25	MAP
Single term index	Query Processing Time (sec)
BM25	0.4313	1.2497

My baseline MAP score is relatively high at 0.4313 and executes quickly as well. 

Retrieval Model:
PRF	MAP
Single Term Index	Query Processing Time (sec)
N=20 M=5 num.idf	0.3712	4.3490
N=20 M=5 num.ntf.idf	0.2029	
N=20 M=3 num.idf	0.4108	3.8528
N=20 M=3 num.ntf.idf	0.2936	
N=20 M=2 num.idf	0.3773	3.7114
N=20 M=2 num.ntf.idf	0.2953	
N=20 M=1 num.idf	0.4313	3.3815
N=20 M=1 num.ntf.idf	0.3552	
N=15 M=5 num.idf	0.3469	4.1746
N=15 M=5 num.ntf.idf	0.2521	
N=15 M=3 num.idf	0.3643	3.9765
N=15 M=3 num.ntf.idf	0.2609	
N=15 M=2 num.idf	0.4173	4.1808
N=15 M=2 num.ntf.idf	0.2884	
N=15 M=1 num.idf	0.4313	3.6116
N=15 M=1 num.ntf.idf	0.3223	
N=10 M=5 num.idf	0.3625	4.1747
N=10 M=5 num.ntf.idf	0.2758	
N=10 M=3 num.idf	0.4044	4.5306
N=10 M=3 num.ntf.idf	0.3578	
N=10 M=2 num.idf	0.3506	3.8603
N=10 M=2 num.ntf.idf	0.3237	
N=10 M=1 num.idf	0.4161	3.9007
N=10 M=1 num.ntf.idf	0.3636	
N=5 M=5 num.idf	0.3306	4.2403
N=5 M=5 num.ntf.idf	0.3350	
N=5 M=3 num.idf	0.4194	4.7174
N=5 M=3 num.ntf.idf	0.3807	
N=5 M=2 num.idf	0.4242	4.8850
N=5 M=2 num.ntf.idf	0.3985	
N=5 M=1 num.idf	0.3851	5.3632
N=5 M=1 num.ntf.idf	0.4069	
N=3 M=5 num.idf	0.4315	4.6200
N=3 M=5 num.ntf.idf	0.3710	
N=3 M=3 num.idf	0.4545	4.3524
N=3 M=3 num.ntf.idf	0.3844	
N=3 M=2 num.idf	0.4252	4.1119
N=3 M=2 num.ntf.idf	0.3763	
N=3 M=1 num.idf	0.4303	4.6831
N=3 M=1 num.ntf.idf	0.4123	
N=1 M=5 num.idf	0.4281	4.569
N=1 M=5 num.ntf.idf	0.3826	
N=1 M=3 num.idf	0.4281	4.7622
N=1 M=3 num.ntf.idf	0.3972	
N=1 M=2 num.idf	0.4310	3.6704
N=1 M=2 num.ntf.idf	0.3998	
N=1 M=1 num.idf	0.4313	4.3325
N=1 M=1 num.ntf.idf	0.4042	

For pseudo relevance feedback I decided to do rankings of with every possible set of parameters for N values of 1, 3, 4, 10, 15, and 20 and M values of 1, 2, 3, and 5 and top terms criteria of num.idf and num.ntf.idf. All of the execution times for PRF were much higher than the baseline and the query reduction times. This is because for PRF, the program had to first run the baseline bm25 rankings, then compute the “good terms” and add them to the queries, and then run the bm25 rankings again. So, the longer execution times line up with what would be expected with most of them between 4 and 5 seconds. I coded my program to calculate num.idf and num.ntf.idf each time, so the execution times listed are the combined time that it took to do num.ntf.idf and num.idf for each pair of N and M values. As a whole, the num.idf MAP scores were higher than the num.ntf.idf values. In every set of N and M values, the num.ntf.idf score was lower except for twice for the values N=5 M=5, and N=5 M=1. As N changes, there is a variation in MAP scores, which indicates the sensitivity of PRF performance to the number of top-ranked documents considered for feedback. For example, with M=5, using num.idf, the MAP scores seem to generally increase as N decreases, suggesting that focusing on fewer top documents might be beneficial for relevance feedback in this particular setup. Different values of M also show variation in MAP scores. This indicates that the number of terms used for feedback significantly impacts the performance. For example, for N=20, num.idf has its best performance at M=1, whereas for num.ntf.idf, M=5 seems to perform poorly compared to M=1 or M=3. From these observations, one can infer that for PRF, both the choice of terms and the number of documents and terms considered can significantly impact retrieval effectiveness. To optimize PRF, it may require fine-tuning these parameters based on the characteristics of the dataset and the information needs represented by the queries. The highest MAP score for this dataset was achieved using N=3, M=3, and num.idf and had a score of 0.4545.

Retrieval Model:
Reduction	MAP
Single term index	Query Processing Time (sec)
Reduction (45%)	0.3856	2.9633
Reduction (35%)	0.3910	2.020
Reduction (30%)	0.3975	1.4871
Reduction (25%)	0.4128	1.0826
Reduction (20%)	0.3213	1.0022
Reduction (15%)	0.1771	0.8367

As the percentage of reduction increases, there is a general trend of increasing MAP scores up to a certain point. The highest MAP score is achieved at a 25% reduction level, indicating that reducing the query to this level removes less informative terms while retaining those most likely to retrieve relevant documents. However, as the reduction threshold is further increased beyond 25%, MAP scores decrease, suggesting that too much reduction may eliminate valuable terms from the query, thereby negatively impacting retrieval effectiveness. There is a clear inverse relationship between the percentage of query reduction and the query processing time. As the reduction threshold increases (and the size of the query decreases), the processing time decreases. This is expected since smaller queries require less computational work during retrieval. The optimal balance between effectiveness (as measured by MAP) and efficiency (as measured by processing time) appears to be at the 25% reduction level. This level yields the highest MAP score and still benefits from reduced processing time compared to lower thresholds. The best result of 0.4128 is a little lower than the baseline score, but it was achieved a little faster.

Retrieval Model:
PRF-Reduction	MAP
Single term index	Query Processing Time (sec)
N=3 M=3 num.idf (25%)	0.4545	3.8553
N=10 M=1 num.idf (25%)	0.4128	3.6587
N=20 M=3 num.idf (25%)	0.3452	3.8910
N=5 M=3 num.idf (15%)	0.1825	2.6606
N=5 M=3 num..ntf.idf (15%)	0.1440	2.6606
N=10 M=5 num.idf (35%)	0.3552	6.3320
N=10 M=5 num.ntf.idf (35%)	0.2853	6.3320

I picked out a few different combinations of parameters that I thought would effectively show PRF-Reduction. The most effective threshold for query reduction was 25%, so I decided to use that as the threshold for PRF-Reduction. The first row combined this with the highest MAP score parameters for PRF of N=3, M=3, and num.idf. The results were the exact same MAP score as the PRF that was done on the baseline results but a slightly faster query processing time. The faster time is likely because the query reduction of 25% runs faster than the baseline bm25 ranking. Rows 2 and 3 were using higher document numbers that also had good MAP scores for PRF. Both scores in this method were slightly lower than their scores with just PRF. The PRF score for N=10 M=1 num.idf was 0.4161 and in conjunction with query reduction the result was 0.4128. The PRF score for N=20 M=3 num.idf was 0.4108 and in conjunction with query reduction the result dropped to 0.3452. The rest of the rows follow similar patterns of being slightly lower MAP scores than their only PRF counterparts. Looking at N=5 M=3 num.idf (15%) and N=5 M=3 num.ntf.idf (15%) however, the MAP scores dropped significantly, likely due to the 15% threshold which as seen earlier performed quite poorly. In summary, it does not appear that for this dataset query expansion and reduction performs better than either one on its own, however, with the right parameters it still performs fairly well.
