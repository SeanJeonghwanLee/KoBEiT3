from transformers import XLMRobertaTokenizer, AutoTokenizer

sen_tokenizer = XLMRobertaTokenizer('./beit3.spm')
test_tokenizer = XLMRobertaTokenizer('./smp.model')
# test_tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

ko = '안녕하세요. 아름다운 밤이에요.'
en = 'Hello. Such a beautiful night.'
encodesen_koto = sen_tokenizer.encode(ko)
encodesen_ento = sen_tokenizer.encode(en)

encodetest_koto = test_tokenizer.encode(ko)
encodetest_ento = test_tokenizer.encode(en)


sen_koto = sen_tokenizer.decode(encodesen_koto)
sen_ento = sen_tokenizer.decode(encodesen_ento)

test_koto = test_tokenizer.decode(encodetest_koto)
test_ento = test_tokenizer.decode(encodetest_ento)

print(f'<sentencepiece tokenizer>')
print(f'ko:{sen_koto}')
print(f'en:{sen_ento}')

print(f'<test tokenizer>')
print(f'test_ko:{test_koto}')
print(f'test_en:{test_ento}')