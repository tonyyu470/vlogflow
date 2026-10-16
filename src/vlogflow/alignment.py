from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher


def tokens(text):
    text=unicodedata.normalize('NFKC',text).lower().replace('’',"'")
    small=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
    tens=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
    def number(m):
        n=int(m.group())
        return small[n] if n<20 else (tens[n//10]+(' '+small[n%10] if n%10 else '')) if n<100 else m.group()
    if not re.search(r'[\u4e00-\u9fff]',text): text=re.sub(r'\b\d{1,2}\b',number,text)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u4e00-\u9fff]",text)


def align_scenes(scenes,words,duration,language):
    expected=[]; ranges=[]
    for s in scenes:
        start=len(expected); expected+=tokens(s['narration']); ranges.append((start,len(expected)))
    observed=[]; timings=[]
    for w in words:
        parts=tokens(w['word'])
        if not math.isfinite(w['start']) or not math.isfinite(w['end']) or w['start']<0 or w['end']<w['start'] or w['end']>duration+.2:
            raise ValueError('Invalid speech recognition timestamps')
        for i,part in enumerate(parts):
            observed.append(part); timings.append((w['start']+(w['end']-w['start'])*i/len(parts),w['start']+(w['end']-w['start'])*(i+1)/len(parts)))
    matching={}
    for a,b,size in SequenceMatcher(None,expected,observed,autojunk=False).get_matching_blocks():
        for k in range(size): matching[a+k]=timings[b+k]
    if not expected or len(matching)/len(expected)<.8 or len(observed)>len(expected)*1.5+2:
        raise ValueError('Generated speech recognition does not match the approved transcript closely enough. Audition the cached voice, improve the reference or shorten the script and retry.')
    spans=[]
    for scene,(a,b) in zip(scenes,ranges):
        found=[matching[i] for i in range(a,b) if i in matching]
        if not found or len(found)/max(1,b-a)<.6:
            raise ValueError(f"Scene {scene['id']}: insufficient transcript alignment")
        spans.append(dict(start=found[0][0],end=found[-1][1],coverage=len(found)/max(1,b-a),words=found))
    result=[]
    for i,span in enumerate(spans):
        start=max(0,span['start']-.16) if i==0 else (spans[i-1]['end']+span['start'])/2
        end=min(duration,span['end']+.22) if i==len(spans)-1 else (span['end']+spans[i+1]['start'])/2
        result.append(dict(scene_id=scenes[i]['id'],start=start,end=end,coverage=span['coverage'],words=span['words']))
    return result


def _chunks(text,n,language):
    parts=list(text) if language=='zh' else text.split()
    # A translation can have fewer words than the source; reduce cue count upstream.
    return [('' if language=='zh' else ' ').join(parts[round(i*len(parts)/n):round((i+1)*len(parts)/n)]).strip() for i in range(n)]


def caption_cues(scenes,voice_result,translations,language,mode):
    cues=[]; offset=0
    for scene,segment in zip(scenes,voice_result['segments']):
        texts=dict(translations.get(scene['id'],{})); texts[language]=scene['narration']
        required=('zh','en') if mode=='bilingual' else (mode,)
        if any(not texts.get(k,'').strip() for k in required): raise ValueError('Missing subtitle translation; generate or save translations first')
        counts={k:len(list(texts[k]) if k=='zh' else texts[k].split()) for k in required}
        n=max(1,max(math.ceil(v/(20 if k=='zh' else 9)) for k,v in counts.items()))
        n=min(n,min(counts.values()))
        chunks={k:_chunks(texts[k],n,k) for k in required}
        duration=segment['duration']; frames_duration=math.ceil(duration*30-1e-7)/30
        stamps=segment.get('words',[])
        # Use recognized word boundaries where available; otherwise whole-scene timing is explicit.
        boundaries=[0]
        for i in range(1,n):
            idx=min(len(stamps)-1,max(0,round(i*len(stamps)/n)))
            boundaries.append(min(duration,max(boundaries[-1]+.01,stamps[idx][0] if stamps else duration*i/n)))
        boundaries.append(duration)
        for i in range(n):
            cues.append(dict(start=offset+boundaries[i],end=offset+boundaries[i+1],**{k:chunks[k][i] for k in required}))
        offset+=frames_duration
    return cues
