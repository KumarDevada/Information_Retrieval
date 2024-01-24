This is my Information retrieval project on the topic Historical information related search based on wikipedia.

sir/madam ,please follow the below mentioned steps to run my project.

* importing libraries : 
1. pip install gensim
2. pip install flask


* Download the dataset(zip file) from : https://drive.google.com/file/d/1q-Rww1zDLzt76oNK27XCVPWMux-qlbcj/view?usp=sharing
  place the downloaded zip file in the folder namely source_code


* Running the project

1. open the folder source_code
2. you can see a zip file containing all the text documents, extract the zip file here.
3. now you can see a folder namely data, containing all the txt documents.
4. now in the folder source code, open the folder named as 'src'
5. folder 'src' has 2 files, namely index.py and search.py.
6. open your terminal inside this directory named as 'src'.
7. now run the command python index.py to start creating inverted indices.


note :  i already executed the command before and generated the folder named as index
        this folder contains 4 files namely dictionary.pkl, inverted_index.pkl, similarity_index.pkl, similarity_index.json 
        these files are responsible for retrieving the relevant documents based on the query.

        after you run the command ( python index.py ) these 4 files will be created.

8. now after the completion of step 7 , you can see the completion log statements in the terminal.
9. inverted index is created in the folder namely index


* viewing the results : there are 2 ways to view the results :
                        1. command line interface
                        2. simple html web page ( recommended )


    * steps to open command line interface

    1. navigate to the directory named as 'src'
    2. open terminal in this directory
    3. run the command python search.py
    4. you can give your query and it shows the relevant document ids and similarity coefficient
    5. type quit to exit from this command line interface.




    * steps to open HTML web page UI

    1. navigate to the root directory namely source_code
    2. open the terminal in this directory
    3. run the command python app.py
    4. you can see that server running on the port number 5000
    5. click ctrl + link to open the webpage or,
    6. type http://127.0.0.1:5000 in the web browser
    7. now you can enter the query and wait for time to get the results.
    8. after getting the results you can click on them to download the correspondong txt file.
    9. you can also see the benchmark time taken for the retrieval process.


