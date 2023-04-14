def array(*a): pass


# BINARY
########

# GPT-2 Small - RESULTS BY FILE - NO ES

# limit-512.chunk-32.epochs-2.batch-16
{'0': {'accuracy': 0.9824399260628466, 'f1': [0.9372937293729373, 0.9897904352498657]},
 '1': {'accuracy': 0.9953789279112755, 'f1': [0.9897750511247444, 0.9970149253731343]},
 '2': {'accuracy': 0.9990757855822551, 'f1': [0.9982964224872232, 0.9993658845909955]},
 '3': {'accuracy': 0.9972273567467652, 'f1': [0.967032967032967, 0.9985528219971056]},
 '4': {'accuracy': 0.9972273567467652, 'f1': [0.4, 0.998610467809171]},
 '5': {'accuracy': 0.9972273567467652, 'f1': [0.9795918367346939, 0.9985126425384234]},
 '6': {'accuracy': 0.9972273567467652, 'f1': [0.961038961038961, 0.9985625299472928]},
 '7': {'accuracy': 0.9842883548983364, 'f1': [0.925764192139738, 0.9912144702842377]},
 '8': {'accuracy': 0.9981515711645101, 'f1': [0.9900990099009901, 0.9989806320081549]}}

# limit-1024.chunk-32.epochs-2.batch-16
{'0': {'accuracy': 0.976915974145891,  'f1': [0.9146757679180887, 0.9866524292578751]},
 '1': {'accuracy': 0.997229916897507,  'f1': [0.9938900203665988, 0.9982089552238806]},
 '2': {'accuracy': 1.0,                'f1': [1.0, 1.0]},
 '3': {'accuracy': 0.9981532779316713, 'f1': [0.9782608695652174, 0.9990356798457087]},
 '4': {'accuracy': 0.9963065558633426, 'f1': [0.0, 0.9981498612395929]},
 '5': {'accuracy': 0.9981532779316713, 'f1': [0.9864864864864865, 0.9990089197224975]},
 '6': {'accuracy': 0.997229916897507,  'f1': [0.96, 0.9985652797704447]},
 '7': {'accuracy': 0.989843028624192,  'f1': [0.9531914893617022, 0.9943034697048162]},
 '8': {'accuracy': 0.9990766389658357, 'f1': [0.9950738916256158, 0.999490575649516]}}

{'0': {'accuracy': 0.976915974145891,  'f1': [0.9146757679180887, 0.9866524292578751]},
 '1': {'accuracy': 0.997229916897507,  'f1': [0.9938900203665988, 0.9982089552238806]},
 '2': {'accuracy': 1.0,                'f1': [1.0, 1.0]},
 '3': {'accuracy': 0.9981532779316713, 'f1': [0.9782608695652174, 0.9990356798457087]},
 '4': {'accuracy': 0.9963065558633426, 'f1': [0.0, 0.9981498612395929]},
 '5': {'accuracy': 0.9981532779316713, 'f1': [0.9864864864864865, 0.9990089197224975]},
 '6': {'accuracy': 0.997229916897507,  'f1': [0.96, 0.9985652797704447]},
 '7': {'accuracy': 0.989843028624192,  'f1': [0.9531914893617022, 0.9943034697048162]},
 '8': {'accuracy': 0.9990766389658357, 'f1': [0.9950738916256158, 0.999490575649516]}}

# limit-2048.chunk-32.epochs-2.batch-16
{'4': {'accuracy': 0.997229916897507, 'f1': [0.4, 0.998611753817677]}}

# limit-10240.chunk-32.epochs-2.batch-160
{'4': {'accuracy': 0.9981532779316713,'f1': [0.6666666666666666, 0.9990740740740741]}}


# GPT-2 Medium
{'4': {'accuracy': 0.9963065558633426, 'f1': [0.0, 0.9981498612395929]}}


# MULTI
########

# limit-512.chunk-32.epochs-2.batch-16
[[139  10   2   2   1   0   0   0   0]
 [  1 245   0   0   0   0   0   0   0]
 [  0 102 191   1   0   0   0   0   0]
 [  3   0  36   8   0   0   0   0   0]
 [  0   0   0   2   0   1   1   0   0]
 [  0   1   4  15  47   8   0   0   0]
 [  0   0   1   1  22  14   1   0   0]
 [  4   3   1   3   1   7 102   1   0]
 [  0   0   2   0   5  12  55  27   0]]
{'all': {'accuracy': 0.9721518987341772, 'f1': [0.9619377162629758, 0.9780439121756487]}}
