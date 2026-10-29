import { useCallback, useState } from 'react';

/** Sentences to read aloud while recording: original, everyday, phonetically varied, 6–12 words. */
export const PROMPTS = {
  en: [
    'The yellow bus stopped right outside the old library.',
    'Please bring two cups of hot tea to the meeting room.',
    'My brother fixed the broken bicycle chain on Sunday morning.',
    'Fresh mangoes from the market taste sweeter in early summer.',
    'Quietly close the window before the evening rain begins.',
    'She painted the garden fence a bright shade of green.',
    'We watched the cricket match until the very last ball.',
    'The phone battery died just as the call was getting interesting.',
    'Grandmother told us a funny story about her village school.',
    'Turn left at the temple and walk straight past the bakery.',
    'Seven little ducks followed their mother across the muddy pond.',
    'Thick clouds covered the hills, so the flight was delayed.',
    'Remember to water the tulsi plant on the balcony daily.',
    'The new office printer jams whenever someone prints in colour.',
    'Bright fireworks lit up the whole sky during the festival.',
  ],
  hi: [
    'पीली बस पुरानी लाइब्रेरी के ठीक सामने रुकी।',
    'मीटिंग रूम में दो कप गरम चाय ले आइए।',
    'मेरे भाई ने रविवार सुबह साइकिल की चेन ठीक की।',
    'बाज़ार के ताज़े आम गर्मियों में और मीठे लगते हैं।',
    'शाम की बारिश से पहले खिड़की धीरे से बंद कर दो।',
    'उसने बगीचे की बाड़ को चमकीले हरे रंग से रंगा।',
    'हमने आख़िरी गेंद तक क्रिकेट मैच देखा।',
    'बातचीत दिलचस्प हुई ही थी कि फ़ोन की बैटरी ख़त्म हो गई।',
    'दादी ने अपने गाँव के स्कूल की मज़ेदार कहानी सुनाई।',
    'मंदिर से बाएँ मुड़कर बेकरी के आगे सीधे चलिए।',
    'सात छोटी बत्तखें अपनी माँ के पीछे तालाब पार कर गईं।',
    'पहाड़ों पर घने बादल थे, इसलिए उड़ान देर से गई।',
    'बालकनी में तुलसी के पौधे को रोज़ पानी देना याद रखना।',
    'नया प्रिंटर रंगीन प्रिंट करते ही अटक जाता है।',
    'त्योहार में आतिशबाज़ी से पूरा आसमान जगमगा उठा।',
  ],
  hinglish: [
    'Peeli bus purani library ke bilkul saamne ruki.',
    'Meeting room mein do cup garam chai le aao please.',
    'Mere bhai ne Sunday subah cycle ki chain theek ki.',
    'Market ke taaze aam garmi mein aur meethe lagte hain.',
    'Shaam ki baarish se pehle khidki dheere se band kar do.',
    'Usne garden ki fence ko chamkeele hare rang se paint kiya.',
    'Humne last ball tak cricket match dekha.',
    'Baat interesting hui hi thi ki phone ki battery khatam ho gayi.',
    'Dadi ne apne gaon ke school ki mazedaar kahani sunayi.',
    'Mandir se left mudkar bakery ke aage seedha chalo.',
    'Saat chhoti battakhen apni maa ke peeche talaab paar kar gayin.',
    'Pahadon par ghane baadal the, isliye flight late hui.',
    'Balcony mein tulsi ke paudhe ko roz paani dena yaad rakhna.',
    'Naya printer colour print karte hi atak jaata hai.',
    'Tyohaar mein aatishbaazi se poora aasmaan jagmaga utha.',
  ],
} as const;
export type PromptLang = keyof typeof PROMPTS;
export const LANG_LABEL: Record<PromptLang, string> = { en: 'English', hi: 'हिन्दी', hinglish: 'Hinglish' };

const key = (pid: string) => `automl.speakerPromptLang.${pid}`;
const readLang = (pid: string): PromptLang => {
  try {
    const v = localStorage.getItem(key(pid));
    return v === 'hi' || v === 'hinglish' ? v : 'en';
  } catch {
    return 'en';
  }
};

/** A sentence to read: the project phrase if set, else a rotating built-in prompt in the chosen language. */
export function usePrompt(projectId: string, phrase?: string | null) {
  const [lang, setLangState] = useState<PromptLang>(() => readLang(projectId));
  const [i, setI] = useState(() => Math.floor(Math.random() * PROMPTS.en.length));
  const setLang = useCallback((l: PromptLang) => {
    setLangState(l);
    try { localStorage.setItem(key(projectId), l); } catch { /* storage unavailable: per-visit only */ }
  }, [projectId]);
  /** Different sentence each call (never the same twice in a row). */
  const next = useCallback(() => setI(x => (x + 1 + Math.floor(Math.random() * (PROMPTS.en.length - 1))) % PROMPTS.en.length), []);
  return { text: phrase || PROMPTS[lang][i]!, fromPhrase: !!phrase, lang, setLang, next };
}
