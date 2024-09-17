from typing import List, Dict, Union, Any, Tuple, Literal
import os
import json
import re
from word2number.w2n import word_to_num as english_word2int
import torch
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from src.models.base import BaseModel
from src.enum import T5DomainClassTypes, T5PriceSubclassTypes
from src.misc.schemas import (
    PriceExtractionSchema,
    ProductNamedEntityExtractionSchema,
    SpecsExtractionSchema,
    SpecificationSchema,
)


def arabic_word2int(textnum, numwords={}):
    if not numwords:
        units = [
            "",
            "واحد",
            "اثنان",
            "ثلاثة",
            "أربعة",
            "خمسة",
            "ستة",
            "سبعة",
            "ثمانية",
            "تسعة",
            "عشرة",
            "أحد عشر",
            "اثنا عشر",
            "ثلاثة عشر",
            "أربعة عشر",
            "خمسة عشر",
            "ستة عشر",
            "سبعة عشر",
            "ثمانية عشر",
            "تسعة عشر",
        ]

        tens = [
            "عشرون",
            "ثلاثون",
            "أربعون",
            "خمسون",
            "ستون",
            "سبعون",
            "ثمانون",
            "تسعون",
        ]

        scales = ["مية", "الف", "مليون", "مليار", "ترليون"]

        numwords["و"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, (idx + 2) * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current


ARABIC_TEXT_PATTERN = r"[\u0600-\u06ff]|[\u0750-\u077f]|[\ufb50-\ufc3f]|[\ufe70-\ufefc]"


class T5NERModel(BaseModel):
    def __init__(
        self,
        model_ckpt_dir: str = os.path.join("ckpt", "ner_model"),
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        super().__init__()
        self.domains = [
            line.replace("\n", "")
            for line in open(
                os.path.join(model_ckpt_dir, "domain_special_tokens.txt"), "r"
            ).readlines()
            if "[" in line and "]" in line
        ]
        self.slots = [
            line.replace("\n", "")
            for line in open(
                os.path.join(model_ckpt_dir, "slot_special_tokens.txt"), "r"
            ).readlines()
            if "[" not in line or "]" not in line
        ]
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt_dir)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(model_ckpt_dir)
        self.t5config = T5Config.from_pretrained(model_ckpt_dir)
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(
            ["<sos_context>"]
        )[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(
            ["<eos_context>"]
        )[0]

        bos_token_id = self.t5config.decoder_start_token_id
        eos_token_id = self.tokenizer.eos_token_id
        # bos_token = self.tokenizer.convert_ids_to_tokens([bos_token_id])[0]
        # eos_token = self.tokenizer.convert_ids_to_tokens([eos_token_id])[0]
        # pad_token_id = self.tokenizer.convert_tokens_to_ids(["<_PAD_>"])[0]
        all_sos_token_list = ["<sos_b>", "<sos_a>", "<sos_r>"]
        all_eos_token_list = ["<eos_b>", "<eos_a>", "<eos_r>"]
        # print(pipe)
        self.special_token_list = [
            "<_PAD_>",
            "<go_r>",
            "<go_b>",
            "<go_a>",
            "<eos_u>",
            "<eos_r>",
            "<eos_b>",
            "<eos_a>",
            "<go_d>",
            "<eos_d>",
            "<sos_u>",
            "<sos_r>",
            "<sos_b>",
            "<sos_a>",
            "<sos_d>",
            "<sos_db>",
            "<eos_db>",
            "<sos_context>",
            "<eos_context>",
        ]

        all_sos_token_id_list = []
        for token in all_sos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            all_sos_token_id_list.append(one_id)
        all_eos_token_id_list = []
        for token in all_eos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            all_eos_token_id_list.append(one_id)

        self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("translate dialogue to belief state:")
        )

    def predict(
        self, input_query, *args: Any, **kwds: Any
    ) -> ProductNamedEntityExtractionSchema:
        extraction_results = ProductNamedEntityExtractionSchema()
        parsed_entities_pred = self.pipe(input_query)
        print(parsed_entities_pred)
        if len(parsed_entities_pred.keys()):
            if T5DomainClassTypes.BRAND.value in parsed_entities_pred.keys():
                extraction_results.brand_extraction = parsed_entities_pred[
                    T5DomainClassTypes.BRAND.value
                ]
            if T5DomainClassTypes.PRICE.value in parsed_entities_pred.keys():
                price_extraction_result: PriceExtractionSchema = PriceExtractionSchema()
                extraction_results.price_extraction = price_extraction_result
                price_entities_pred = parsed_entities_pred[
                    T5DomainClassTypes.PRICE.value
                ]
                is_upper_parsed = True
                is_lower_parsed = True
                if T5PriceSubclassTypes.CURRENCY.value in price_entities_pred.keys():
                    price_extraction_result.unit = price_entities_pred[
                        T5PriceSubclassTypes.CURRENCY.value
                    ]
                if (
                    T5PriceSubclassTypes.GE.value in price_entities_pred.keys()
                    and T5PriceSubclassTypes.LE.value in price_entities_pred.keys()
                ):
                    upper_range = price_entities_pred[T5PriceSubclassTypes.LE.value]
                    lower_range = price_entities_pred[T5PriceSubclassTypes.GE.value]

                    try:
                        upper_range = float(upper_range)
                    except:
                        try:
                            if re.match(
                                pattern=ARABIC_TEXT_PATTERN, string=input_query
                            ):
                                upper_range = float(arabic_word2int(upper_range))
                            else:
                                upper_range = float(english_word2int(upper_range))
                        except:
                            is_upper_parsed = False

                    try:
                        lower_range = float(lower_range)
                    except:
                        try:
                            if re.match(
                                pattern=ARABIC_TEXT_PATTERN, string=input_query
                            ):
                                lower_range = float(arabic_word2int(lower_range))
                            else:
                                lower_range = float(english_word2int(lower_range))
                        except:
                            is_lower_parsed = False
                    if is_lower_parsed and is_upper_parsed:
                        price_extraction_result.upper_range = max(
                            upper_range, lower_range
                        )
                        price_extraction_result.lower_range = min(
                            lower_range, upper_range
                        )
                    elif is_upper_parsed:
                        price_extraction_result.upper_range = upper_range
                    elif is_lower_parsed:
                        price_extraction_result.lower_range = lower_range
                elif T5PriceSubclassTypes.GE.value in price_entities_pred.keys():
                    lower_range = price_entities_pred[T5PriceSubclassTypes.GE.value]

                    try:
                        lower_range = float(lower_range)
                    except:
                        try:
                            if re.match(
                                pattern=ARABIC_TEXT_PATTERN, string=input_query
                            ):
                                lower_range = float(arabic_word2int(lower_range))
                            else:
                                lower_range = float(english_word2int(lower_range))
                        except:
                            is_lower_parsed = False
                    if is_lower_parsed:
                        price_extraction_result.lower_range = lower_range
                elif T5PriceSubclassTypes.LE.value in price_entities_pred.keys():
                    upper_range = price_entities_pred[T5PriceSubclassTypes.LE.value]
                    try:
                        upper_range = float(upper_range)
                    except:
                        try:
                            if re.match(
                                pattern=ARABIC_TEXT_PATTERN, string=input_query
                            ):
                                upper_range = float(arabic_word2int(upper_range))
                            else:
                                upper_range = float(english_word2int(upper_range))
                        except:
                            is_upper_parsed = False
                    if is_upper_parsed:
                        price_extraction_result.upper_range = upper_range
                elif T5PriceSubclassTypes.EQ.value in price_entities_pred.keys():
                    exact_range = price_entities_pred[T5PriceSubclassTypes.EQ.value]
                    try:
                        exact_range = float(exact_range)
                    except:
                        try:
                            if re.match(
                                pattern=ARABIC_TEXT_PATTERN, string=input_query
                            ):
                                exact_range = float(arabic_word2int(exact_range))
                            else:
                                exact_range = float(english_word2int(exact_range))
                        except:
                            is_lower_parsed = False
                    if is_lower_parsed:
                        price_extraction_result.upper_range = exact_range
                        price_extraction_result.lower_range = exact_range
            if T5DomainClassTypes.SPECS.value in parsed_entities_pred.keys():
                specs_extraction_result: SpecsExtractionSchema = SpecsExtractionSchema()
                extraction_results.specs_extraction = specs_extraction_result
                specs_entities_pred = parsed_entities_pred[
                    T5DomainClassTypes.SPECS.value
                ]
                for spec_key, spec_val in specs_entities_pred.items():
                    specs_extraction_result.specs.append(
                        SpecificationSchema(spec_name=spec_key, spec_val=spec_val)
                    )
            if T5DomainClassTypes.RATE.value in parsed_entities_pred.keys():
                extraction_results.rate_extraction = parsed_entities_pred[
                    T5DomainClassTypes.RATE.value
                ]
            if T5DomainClassTypes.SUBCATEGORY.value in parsed_entities_pred.keys():
                extraction_results.sub_category_extraction = parsed_entities_pred[
                    T5DomainClassTypes.SUBCATEGORY.value
                ]
            if T5DomainClassTypes.SUPERCATEGORY.value in parsed_entities_pred.keys():
                extraction_results.super_category_extraction = parsed_entities_pred[
                    T5DomainClassTypes.SUPERCATEGORY.value
                ]

        return extraction_results

    def parse_bs(self, sent) -> Dict[str, Union[Dict[str, str], str]]:
        """Convert compacted bs span to triple list
        Ex:
        """
        sent = sent.strip("</s>")
        sent = sent.split()
        belief_state = {}
        domain_idx = [idx for idx, token in enumerate(sent) if token in self.domains]
        for i, d_idx in enumerate(domain_idx):
            next_d_idx = len(sent) if i + 1 == len(domain_idx) else domain_idx[i + 1]
            domain = sent[d_idx]
            sub_span = sent[d_idx + 1 : next_d_idx]
            sub_s_idx = [
                idx for idx, token in enumerate(sub_span) if token in self.slots
            ]
            if len(sub_s_idx) == 0:
                belief_state[domain] = " ".join(sub_span)
            for j, s_idx in enumerate(sub_s_idx):
                next_s_idx = (
                    len(sub_span) if j == len(sub_s_idx) - 1 else sub_s_idx[j + 1]
                )
                slot = sub_span[s_idx]
                value = " ".join(sub_span[s_idx + 1 : next_s_idx])
                bs = " ".join([value])
                if domain not in belief_state.keys():
                    belief_state[domain] = {}
                belief_state[domain][slot] = bs
        return belief_state

    def pipe(self, text):
        text = f"<sos_u> {text} <eos_u>"
        # print(tokenizer.tokenize(sent))
        max_decode_len = 120
        (
            pad_token_id,
            start_token_id,
            end_token_id,
        ) = self.tokenizer.convert_tokens_to_ids(
            [
                "<_PAD_>",
                "<sos_b>",
                "<eos_b>",
            ]
        )
        src_input = torch.LongTensor(
            [
                self.bs_prefix_id
                + [self.sos_context_token_id]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                + [self.eos_context_token_id]
            ]
        )
        src_mask = torch.ones_like(src_input)
        src_mask = src_mask.masked_fill(src_input.eq(pad_token_id), 0.0).type(
            torch.FloatTensor
        )
        src_input = src_input.to(self.device)
        src_mask = src_mask.to(self.device)
        start_token, end_token = "<sos_b>", "<eos_b>"
        outputs = self.model.generate(
            input_ids=src_input,
            attention_mask=src_mask,
            decoder_start_token_id=start_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=end_token_id,
            max_length=max_decode_len,
        )

        one_res_text = self.tokenized_decode(outputs[0])
        one_res_text = one_res_text.split(start_token)[-1].split(end_token)[0].strip()

        final_res_list = []
        for token in one_res_text.split():
            if token == "<_PAD_>":
                continue
            else:
                final_res_list.append(token)
        one_res_text = " ".join(final_res_list).strip()

        return self.parse_bs(one_res_text)

    def tokenized_decode(self, token_id_list):
        pred_tokens = self.tokenizer.convert_ids_to_tokens(token_id_list)
        res_text = ""
        curr_list = []
        for token in pred_tokens:
            if token in self.special_token_list + ["<s>", "</s>", "<pad>"]:
                if len(curr_list) == 0:
                    res_text += " " + token + " "
                else:
                    curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
                    res_text = res_text + " " + curr_res + " " + token + " "
                    curr_list = []
            else:
                curr_list.append(token)
        if len(curr_list) > 0:
            curr_res = self.tokenizer.convert_tokens_to_string(curr_list)
            res_text = res_text + " " + curr_res + " "
        res_text_list = res_text.strip().split()
        res_text = " ".join(res_text_list).strip()
        return res_text


if __name__ == "__main__":
    model = T5NERModel(model_ckpt_dir=r"ckpt\ner_model")

    print(
        json.dumps(
            model(input_query="يساوي 12 ريال و عشرون ريال").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="من 12 ريال").model_dump(), indent=3, ensure_ascii=False
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="أقل 12 ريال").model_dump(), indent=3, ensure_ascii=False
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="أكثر من 12 ريال").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="أكثر من 12 إلى 20 ريال").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(input_query="عطر ديور ارخص من ٢٠٠").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("=====================")
    print(
        json.dumps(
            model(
                input_query="عطر ديور ارخص من ٢٠٠ حجمه 12 ميليلتر و تقيمه 13"
            ).model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
    print("======================")
    print(
        json.dumps(
            model(input_query="Backpack Herschel, Black - 70 SAR").model_dump(),
            indent=3,
            ensure_ascii=False,
        )
    )
