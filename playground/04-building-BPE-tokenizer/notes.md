**Get stats function**
To start we need to encode the text to bytes to list of integers and then go through every pair of integers, counting the frequence of each pair, this function will be very usefull to train the tokenizer.

**Merge function**
Then you have to create a merge function, it takes the tokenized dataset, the pair you want to merge and the new id that will replace that pair, on a while loop you have to check if the current pair (current token, current token + 1) matches the pair to be replaced, also check to avoid index overflow, if it matches save it and increase the index +2 since we replaced not only the current token but the next too, otherwise just save the token, at the end return all tokens.

**Tokenizer training**
To train the tokenizer you need a loop, first get the stats of the corpus, get the most common pair, replace the most common pair with a new id in the corpus, save the merge, repeat the loop updating the same corpus to get bigger merges over time.

**Decoding stage**
So for decoding is simple, we need to create a vocab dictionary that holds the bytes for each token id generated in the merges, at first I thought of a recursive approach, however since the merges are already sorted we can just iterate over each merge sequencially, so each merge will use valid bytes merged previously in the same vocabulary dictionary. Then you just have to get the bytes from each token id and decode it as utf-8

**Encoding stage**
The algorithm is simple, we just go over every pair of bytes from the text to encode, and check in merges what is the most viable pair to be a merge, you want to get the lowest possible to merge in order, so you replace the pair with the merge id and repeat until there are not more valid merges avaiable in the sequence.



