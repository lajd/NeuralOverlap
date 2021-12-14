"""
# module Words

- Julia version: 
- Author: jon
- Date: 2021-12-12

# Examples

```jldoctest
julia>
```
"""

module Words
    function getWords(wordFilePath="/Users/jonathan/PycharmProjects/NeuralOverlap/data/experiments/raw_data/word", training=false, eval=false, train_ratio=0.7)::Array{String}
        words = readlines(wordFilePath)
        return words
    end
end
