#!/usr/bin/python
#
# to use:
#import sys
#sys.path.append("path_to_OneScan")
# 
#import OneScan
#...
#tokens = OneScan.oneScanTokenizer(text)
#...

#one scan tokenizer
def oneScanTokenizer(text):
	"""
	This tokenizer adopted the rules in the implementation of nltk's TreebankWordTokenizer in
	Python port by edward Loper and Michael Heilman (available at 
	http://www.nltk.org/_modules/nltk/tokenize/treebank.html).

	This tokenizer is implemented in one scan method in which tokens are splitted on fly
	during one scan. it performs the following steps:

	1. split standard contractions, e.g. ``don't`` => ``do n't`` and ``they'll`` => ``they 'll``;
	2. treat most punctuation characters as separate tokens;
	3. split off commas and single quotes, when followed by whitespace;
	4. separate periods that appear at the end of line.

	>>> from OneScan import oneScanTokenizer
	>>> s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
	>>> oneScanTokenizer(s)
	['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 
	'of', 'them.', 'Thanks', '.']
	>>> s = "They'll save and invest more."
	>>> oneScanTokenizer(s)
	['They', "'ll", 'save', 'and', 'invest', 'more', '.']
	>>> s = "hi, my name can't hello,"
	>>> oneScanTokenizer(s)
	['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']

	"""

	len1 = len(text)
	#avoid text[end +10] out of range    
	text += '          '
	rule1 = 'sSmMdD'
	rule2 = ["'ll","'LL","'re","'RE","'ve","'VE","n't","N'T"] 
	#whitespace, non-alphanumeric, non-underscore character
#	boundary = ' \t\n\r\f\v!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
	boundary = '\t\n\x0b\x0c\r !"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
#	delimiter = ' \t\n\r\f\v":,.;@#$%&?![](){}<>\'-'
	changed_lst = ' !"#$%&\'(),-:;<>?@[]{}'
	lst = []
	start = 0
	end = 0
	prev = ' '
	updated = 0

	#scan 'text' by pointer 'end'
	while end < len1:
		c =  text[end]
        
		#move to next char if not split
		if c not in boundary :			
			prev = c 
			end += 1 
			continue

		#check contrations first before losed the first '\b' in constractions        
		prev, start, end, lst, updated = contractions(prev, c, start, end, text, lst, boundary)
		if updated == 1:
			continue
		        
		if c in ' \t\n\r\f\v':
			#split into token if not empty
			if end > start:
				lst += [text[start:end]] 
			prev = c
			end += 1
			start = end
			continue
		    
		elif c == '`' and text[end+1] == '`':
			#(1.2) punctuation 
			#text = re.sub(r'(``)', r' \1 ', text)
			if end > start:
				lst += [text[start:end]] 
			lst += ["``"]
			prev = '`'
			end += 2
			start = end
			continue             

		elif c == '"':
			if prev in ' ([{<`':
				#(1.1) starting quotes  - since prev is ' ' for the first char
				#s(1.3) tarting quotes 
				#text = re.sub(r'([ (\[{<])"', r'\1 `` ', text)
				if end > start:
					lst += [text[start:end]]               
				lst += ["``"]
				prev = c
				end += 1
				start = end
				continue

			else:
				#(4.1) ending quotes             
				#text = re.sub(r'"', " '' ", text)
				if end > start:
					lst += [text[start:end]]            
				lst += ["''"]
				prev = c
				end += 1
				start = end
				continue

		elif c == "," or c in ":" :
			if not text[end+1].isdigit() :
				if (text[end -1] != c or (text[end -1] == c and text[end-2] == c)):
					#(2.1) punctuation 
					#text = re.sub(r'([:,])([^\d])', r' \1 \2', text)
					if end > start:
						lst += [text[start:end]]           
					lst += [c]
					prev = c
					end += 1
					start = end
					continue 		
		elif c == '.':
			if text[end+1] == '.' and text[end+2] == '.' :
				#(2.2) punctuation 
				#text = re.sub(r'\.\.\.', r' ... ', text)
				if end > start:
				    lst += [text[start:end]] 
				lst += ["..."]
				prev = '.'
				end += 3
				start = end
				continue

			elif end+1 == len1:                
			#elif prev != '.' and not prev.isdigit() and text[end+1] not in '])}>"\'' and end+1 == len1:
				#(2.4) punctuation is not implemeted correctly
				#text = re.sub(r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', text)               
				if end > start:
				    lst += [text[start:end]]          
				lst += [c]
				prev = c
				end += 1
				start = end
				continue

		elif c in ';@#$%&?![](){}<>':
			#(2.3) and (2.5) punctuation             
			#text = re.sub(r'[;@#$%&]', r' \g<0> ', text)
			#text = re.sub(r'[?!]', r' \g<0> ', text)
			#(3.1) parens and brackets             
			#text = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', text)
			if end > start:
				lst += [text[start:end]]          
			lst += [c]
			prev = c
			end += 1
			start = end
			continue 

		elif c == '-':
			if text[end+1] == '-':
				#parens and brackets (3.2)            
				#text = re.sub(r'--', r' -- ', text)
				if end > start:
					lst += [text[start:end]]            
				lst += ['--']
				prev = c
				end += 2
				start = end
				continue 

		elif c == "'" :
			next1 = text[end+1]
			next2 = text[end+2]
			next3 = text[end+3]
			if  next1 == "'":
				if prev not in ' \t\n\r\f\v':
					#(4.2) ending quotes             
					#text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)
					if end > start:
						lst += [text[start:end]]            
					lst += ["''"]
					prev = c
					end += 2
					start = end
					continue
			elif prev != "'":
				if (next1 in ' ":,;@#$%&?![](){}<>')\
					or (next1 == '-' and next2 == '-')\
					or (next1 == '.' and next2 == '.' and next3 == '.') : 
					#(2.6) punctuation 
					#text = re.sub(r"([^'])' ", r"\1 ' ", text)
					if end > start:
						lst += [text[start:end]]            
					lst += [c]
					prev = c
					end += 1
					start = end
					continue 
			if next1 in rule1:
				if next2 in ' ":,;@#$%&?![](){}<>'\
					or (next2 == '-' and next3 == '-')\
					or (next2 == '.' and next3 == '.' and next3 == '.') : 
					#(4.3) reules 1 
					#text = re.sub(r"([^' ])('[sS]|'[mM]|'[dD]|') ", r"\1 \2 ", text)
					if end > start:
						lst += [text[start:end]] 
					lst += ["'" + next1]
					prev = next1
					end += 2
					start = end
					continue			    
			elif prev.lower() == 'n' and next1.lower() == 't':
				if next2 in ' ":,;@#$%&?![](){}<>'\
					or (text[end+2:end+4] == '--')\
					or (text[end+2:end+5] == '...') : 
					#(4.4) reules 2 
					#text = re.sub(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", r"\1 \2 ", text)
					if end - 1 > start:
						lst += [text[start:end-1]] 
					lst += [prev + "'" + next1]
					prev = next1
					end += 2
					start = end
					continue
			elif "'" + next1 + next2 in rule2:
				if next3 in ' ":,;@#$%&?![](){}<>'\
					or (text[end+3:end+5] == '--')\
					or (text[end+3:end+6] == '...') :
					#(4.4) reules 2 
					#text = re.sub(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", r"\1 \2 ", text)
					if end > start:
						lst += [text[start:end]] 
					lst += ["'" + next1 + next2]
					prev = next2
					end += 3
					start = end
					continue
			elif (next1 == 't' or next1 == 'T') and prev in changed_lst:
				"""
				CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
						 re.compile(r"(?i) ('t)(was)\b")]
				"""        
				token = text[end+1:end+4]
				b = text[end+4]
				token_low = token.lower()
				if token_low == "tis":
					if b in boundary:
						if end > start:
							lst += [text[start:end]]                 
						lst += ["'"+token[0], token[1:]]
						prev = text[end+3]
						end += 4
						start = end
						continue		
				token = text[end+1:end+5]
				b = text[end+5]
				token_low = token.lower()
				if token_low == "twas":
					if b in boundary:
						if end > start:
							lst += [text[start:end]]
						lst += ["'"+token[0], token[1:]]
						prev = text[end+4]
						end += 5
						start = end
						continue		        
		prev = c
		end += 1
	 
		continue
	#end while

	if end > start:
		last_token = text[start:end]
		if last_token != "" and last_token != " ":
			lst += [last_token]

	return lst


#contraction 2 and 3
def contractions(prev, c, start, end, text, lst, boundary):
	"""
	List of contractions adapted from Robert MacIntyre's tokenizer.
	CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
		 re.compile(r"(?i)\b(d)('ye)\b"),
		 re.compile(r"(?i)\b(gim)(me)\b"),
		 re.compile(r"(?i)\b(gon)(na)\b"),
		 re.compile(r"(?i)\b(got)(ta)\b"),
		 re.compile(r"(?i)\b(lem)(me)\b"),
		 re.compile(r"(?i)\b(mor)('n)\b"),
		 re.compile(r"(?i)\b(wan)(na) ")]
	"""
	if text[end+1] in 'cdglwmCDGLMW':
		token = text[end+1 : end+7]
		b =  text[end+7]
		token_low = token.lower()

		if token_low == 'cannot':
			if b in boundary:
				if end > start:
					lst += [text[start:end]]
				if c not in ' \t\n\r\f\v':
					lst += [c]
				lst += [token[:3], token[3:]]
				prev = text[end+6]
				end += 7
				start = end
				return prev, start, end, lst, 1
        
	
		token = text[end+1 : end+6]
		b =  text[end+6]
		token_low = token.lower()
		if token_low == 'gimme' \
			or token_low == 'gonna' \
			or token_low == 'gotta' \
			or token_low == 'lemme' \
			or token_low == 'wanna' \
			or token_low == "mor'n": 
			if b in boundary:                    
				if c == '-' or c == '+':
					lst += [text[start:end+1]]
				else:
					if end > start:
						lst += [text[start:end]]
					if c not in ' \t\n\r\f\v':
						lst += [c]                  
				lst += [token[:3], token[3:]]               
				prev = text[end+5]
				end += 6
				start = end
				return prev, start, end, lst, 1                   
       
		token = text[end+1 : end+5]
		b =  text[end+5]
		token_low = token.lower()
		if token_low == "d'ye":
			if b in boundary:
				if c == '-' or c == '+':
					lst += [text[start:end+1]]
				else:
					if end > start:
						lst += [text[start:end]]
					if c not in ' \t\n\r\f\v':
						lst += [c]                
				lst += [token[0], token[1:]]
				prev = text[end+4]
				end +=5
				start = end
				return prev, start, end, lst, 1                 
            
	return prev, start, end, lst, 0

from nltk.tokenize import TreebankWordTokenizer
import sys

if __name__ == "__main__":
	s = "Florence --cannot be" #what I cannot acannotb acannot cannot/ caneeenot. gotta\t #'tis *cannot cannot9 cannot."
	print ' '.join(TreebankWordTokenizer().tokenize(s))
	print ' '.join(oneScanTokenizer(s))
	sys.exit(0)
	s = '"<> denotes" I\'ve an in-equa...ti--on\'s ... aa("not equal#$ to AT@T"). '
	print ' '.join(TreebankWordTokenizer().tokenize(s))
	print ' '.join(oneScanTokenizer(s))


	s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
	print TreebankWordTokenizer().tokenize(s)
	print oneScanTokenizer(s)
	#['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']

	s = "They'll save and invest more."
	print TreebankWordTokenizer().tokenize(s)
	print oneScanTokenizer(s)
	#['They', "'ll", 'save', 'and', 'invest', 'more', '.']

	s = "hi, my name can't%thello,"
	print TreebankWordTokenizer().tokenize(s)
	print oneScanTokenizer(s)
	#['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']





