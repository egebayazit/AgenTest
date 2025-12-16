# semantic_filter.py

import re
from pprint import pformat
import logging
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Set
logger = logging.getLogger(__name__)


class SemanticStateFilter:
    """
    Test adımları (step, expected, note) ile UI elementlerini eşleştirir.
    Spatial (Mekansal) yakınlık, Kümülatif Puanlama ve Keyword önceliklendirmesi kullanır.
    """

    # UI'da "kontrol" veya "tıklanabilir nesne" olabilecek kelimeler
    GENERIC_CONTROL_KEYWORDS = {
        "check", "tick", "mark", "box", "toggle", "switch", "radio", 
        "btn", "button", "submit", "ok", "cancel", "apply", 
        "icon", "img", "image", "svg", "path", "vector",
        "input", "edit", "field", "text", "combo", "drop", "list", "select",
        "unknown", "control", "widget", "item", "uiview", "container",
        "m0", "l9", "z", "caret", "arrow", "chevron",
        "dropdown", "menu", "panel", "window", "tab", "label",
        # Navigasyon ve Aksiyon Kelimeleri
        "move", "down", "up", "next", "prev", "back", "forward", "scroll", "nav",
        # Görsel OCR Hataları / Semboller (BURASI EKLENDİ)
        "v", ">", "<", "+", "-", "x", "...",
        "°", "'", "\"", "00", "no", "oo", "lat", "long", "alt","Noo"
    }

    # Stop words: Bu kelimeler TEK BAŞINA aranmaz
    STOP_WORDS = {
        "click", "double", "right", "left", "press", "type", "select", 
        "verify", "check", "wait", "button", "input", "list", "panel", 
        "row", "field", "in", "on", "at", "to", "from", "the", "a", "an", "is",
        "find", "locate", "see", "into", "visible", "showing", "options", "should", "be",
        "here", "search", "next", "value", "enter", "item"
    }

    def __init__(self, row_tolerance: int = 40, x_search_range: int = 600, match_threshold: float = 0.70):
        """
        row_tolerance: 40px (Satır hizasından sapma payı - 'move down' gibi kayık butonlar için)
        x_search_range: 600px (Anchor elementten yatayda ne kadar uzağa bakılacağı)
        """
        self.row_tolerance = row_tolerance
        self.x_search_range = x_search_range
        self.match_threshold = match_threshold

    def _extract_search_terms(self, text_sources: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        Metin kaynaklarından arama terimlerini çıkarır.
        """
        strong_phrases = set()
        weak_keywords = set()
        
        full_text = " ".join([str(t) for t in text_sources if t])
        
        # 1. Tırnak içindeki ifadeler (High Priority)
        quoted_phrases = re.findall(r"['\"](.*?)['\"]", full_text)
        for phrase in quoted_phrases:
            clean_phrase = phrase.strip().lower()
            if len(clean_phrase) > 1:
                strong_phrases.add(clean_phrase)

        # 2. Kelime bazlı analiz (Weak Keywords)
        clean_text = re.sub(r"[^\w\s]", " ", full_text)
        words = clean_text.split()
        
        for word in words:
            w_lower = word.lower()
            if w_lower not in self.STOP_WORDS and len(w_lower) > 2:
                weak_keywords.add(w_lower)
        
        # 3. ÖZEL DURUM: "next to X" pattern'i varsa, X'i strong phrase yap
        # "Click the value next to 'Latitude'" -> "latitude" strong olmalı
        next_to_pattern = re.search(r"next\s+to\s+['\"]?(\w+)['\"]?", full_text, re.IGNORECASE)
        if next_to_pattern:
            target = next_to_pattern.group(1).lower()
            strong_phrases.add(target)
            # Ayrıca weak keywords'den kaldır (strong'a dönüştü)
            weak_keywords.discard(target)
                
        return strong_phrases, weak_keywords

    def _score_element(self, element_name: str, strong_phrases: Set[str], weak_keywords: Set[str]) -> int:
        """
        Element ismine puan verir. KÜMÜLATİF (TOPLAMALI) MANTIK.
        """
        if not element_name: return 0
        name_lower = element_name.lower().strip()
        score = 0

        # 1. Strong Match (Tam İfade) -> Direkt 100
        for phrase in strong_phrases:
            if phrase in name_lower:
                return 100 

        # 2. Weak Match (Kümülatif) -> Her eşleşen kelime puanı artırır.
        # "Message Receiver Type" -> Message(30) + Receiver(30) = 60 Puan.
        # Böylece tırnak içinde olmasa bile güçlü bir aday olur.
        matches_count = 0
        for kw in weak_keywords:
            if kw in name_lower:
                matches_count += 1
            elif len(kw) > 4 and abs(len(kw) - len(name_lower)) <= 2:
                 # Fuzzy check (Sadece uzun kelimeler için)
                if SequenceMatcher(None, kw, name_lower).ratio() >= self.match_threshold:
                    matches_count += 0.8 # Fuzzy match tam puan almaz

        if matches_count > 0:
            # En fazla 95 puana kadar çıkabilir (Strong match'i geçemesin)
            score = min(95, matches_count * 30)

        return int(score)

    def _is_potential_control(self, element: Dict[str, Any]) -> bool:
        """Elementin bir kontrol olup olmadığına bakar."""
        name = str(element.get('name', '')).lower()
        el_type = str(element.get('type', '')).lower()
        
        if len(name) < 2 and el_type in self.GENERIC_CONTROL_KEYWORDS: return True
        
        search_text = f"{name} {el_type}"
        if any(k in search_text for k in self.GENERIC_CONTROL_KEYWORDS):
            return True
            
        if re.search(r"m\d+,", name): return True

        return False

    def filter_elements(self, 
                       all_elements: List[Dict[str, Any]], 
                       test_step: str,
                       expected_result: str,
                       note_to_llm: str) -> List[Dict[str, Any]]:
        
        strong_phrases, weak_keywords = self._extract_search_terms([test_step, expected_result, note_to_llm])
        
        if not strong_phrases and not weak_keywords:
            return all_elements[:60]

        anchors = []
        
        # ADIM 1: Anchor Adaylarını Bul ve Puanla
        for el in all_elements:
            name = str(el.get('name', ''))
            score = self._score_element(name, strong_phrases, weak_keywords)
            if score > 0:
                anchors.append({'el': el, 'score': score})

        final_elements = []
        added_ids = set()

        # NOT: Anchor'ları hemen ekleme, önce spatial expansion yap
        # Çünkü bazı elementler hem anchor hem satellite olabilir
        # Anchorları sadece referans olarak kullan

        # ADIM 2: Spatial Expansion (GÜNCELLENMİŞ VERSİYON)
        if anchors:
            max_score = max(a['score'] for a in anchors)
            # Strong phrase match varsa sadece onları kullan (100 puan olanlar)
            strong_matches = [a for a in anchors if a['score'] == 100]
            if strong_matches:
                best_anchors = [a['el'] for a in strong_matches]
            else:
                # Yoksa en yüksek skordan -50 tolerans
                best_anchors = [a['el'] for a in anchors if a['score'] >= max_score - 50]

            for anchor in best_anchors:
                ac = anchor.get('center', {})
                ax, ay = int(ac.get('x', 0)), int(ac.get('y', 0))
                anchor_name = str(anchor.get('name', ''))
                # print(f"[DEBUG] Anchor: '{anchor_name}' at ({ax}, {ay})")

                for el in all_elements:
                    if id(el) in added_ids: continue

                    ec = el.get('center', {})
                    ex, ey = int(ec.get('x', 0)), int(ec.get('y', 0))
                    el_name = str(el.get('name', ''))

                    y_diff = abs(ey - ay)
                    x_diff = abs(ex - ax)

                    # --- KRİTİK GÜNCELLEME BURASI ---
                    # Satır hizası tutuyorsa (y_diff <= tolerans)
                    if y_diff <= self.row_tolerance:
                        
                        # Durum A: Satellite (Uydu) Mantığı - 250px yakınlık
                        # ÖNEMLI: OCR bozuk olsa bile (Noo"00'00" gibi) listeye al
                        if x_diff <= 250:
                            final_elements.append(el)
                            added_ids.add(id(el))
                        # Durum B: Normal Kontrol Mantığı - 251-600px arası
                        elif x_diff <= self.x_search_range and self._is_potential_control(el):
                            final_elements.append(el)
                            added_ids.add(id(el))
                                                # ---------------------------------
            
            # Spatial expansion tamamlandıktan SONRA anchor'ları ekle
            # Böylece anchor'lar her zaman listede olur
            for item in anchors:
                el = item['el']
                if id(el) not in added_ids:
                    final_elements.append(el)
                    added_ids.add(id(el))
        else:
            return all_elements[:60]

        # ADIM 3: Spatial Deduplication
        final_list = self._remove_distant_duplicates(final_elements, anchors)

        final_list.sort(key=lambda e: (e.get('center', {}).get('y', 0), e.get('center', {}).get('x', 0)))
        
        return final_list
    

    def _remove_distant_duplicates(self, elements: List[Dict[str, Any]], anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        HİBRİT TEMİZLİK:
        Hem 'Heading -> Dropdown' (Yatay Hiza) 
        Hem de 'Player Info -> Hostile' (Dikey Hiza) senaryolarını destekler.
        Elementin Anchor ile hizalı olup olmadığına bakar.
        """
        if len(elements) < 2: return elements

        groups = {}
        for el in elements:
            name = str(el.get('name', '')).strip().lower()
            # ÖNEMLI: Bozuk OCR isimleri (Noo"00'00" gibi) generic keyword gibi görünebilir
            # Ama bunlar duplicate değil, sadece bozuk isimler. Skip etme.
            if not name: 
                continue 
            # Generic keyword kontrolünü tamamen kaldırdık - sadece boş isim kontrolü yap
            if name not in groups: groups[name] = []
            groups[name].append(el)

        ids_to_remove = set()
        
        # Referans noktaları (Anchorlar)
        reference_elements = [a['el'] for a in anchors] if anchors else elements

        for name, candidates in groups.items():
            if len(candidates) < 2: continue

            context_elements = [e for e in reference_elements if str(e.get('name', '')).strip().lower() != name]
            if not context_elements: continue

            candidate_scores = []

            for cand in candidates:
                c_center = cand.get('center', {})
                cx, cy = c_center.get('x', 0), c_center.get('y', 0)
                
                min_alignment_score = float('inf')
                
                for ctx in context_elements:
                    ctx_center = ctx.get('center', {})
                    ctx_x, ctx_y = ctx_center.get('x', 0), ctx_center.get('y', 0)
                    
                    dx = abs(cx - ctx_x)
                    dy = abs(cy - ctx_y)
                    
                    # --- HİBRİT SKORLAMA ---
                    # 1. Row Score: Yatay ilişki (Heading -> Dropdown). Y farkı çok cezalandırılır.
                    # dy * 25: 1px dikey kayma 25px yatay kaymaya bedeldir.
                    row_score = dx + (dy * 25)
                    
                    # 2. Column Score: Dikey ilişki (Player Info -> Hostile). X farkı cezalandırılır.
                    # dx * 5: 1px yatay kayma 5px dikey kaymaya bedeldir (Sütunlar genelde daha geniştir).
                    col_score = dy + (dx * 5)
                    
                    # Hangisi daha iyiyse (daha düşükse) onu al.
                    # Yani element ya aynı satırda olmalı ya da aynı sütunda.
                    best_score = min(row_score, col_score)
                    
                    if best_score < min_alignment_score:
                        min_alignment_score = best_score
                
                candidate_scores.append({'el': cand, 'score': min_alignment_score})

            # En düşük skora (en iyi hizaya) sahip olanı seç
            candidate_scores.sort(key=lambda x: x['score'])
            best_candidate = candidate_scores[0]
            
            for other in candidate_scores[1:]:
                # Eşik değeri biraz esnek tutuyoruz
                threshold = 1000 if anchors else 1200
                
                if other['score'] > best_candidate['score'] + threshold:
                    ids_to_remove.add(id(other['el']))

        return [e for e in elements if id(e) not in ids_to_remove]


    def format_for_llm(self, filtered_elements: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        lines = ["ID | Name | Type | (x,y)"]
        lines.append("-" * 60)
        id_map = {} 
        
        for idx, el in enumerate(filtered_elements):
            sim_id = str(idx + 1)
            center = el.get('center', {})
            x, y = int(center.get('x', 0)), int(center.get('y', 0))
            
            etype = str(el.get('type', 'unk')).replace('container', 'cont').replace('unknown', 'unk')
            name = str(el.get('name', '')).replace('\n', ' ').strip()
            
            if len(name) > 40: name = name[:37] + "..."
            if not name: name = "[NO_NAME]"
            
            line = f"{sim_id} | {name} | {etype} | ({x},{y})"
            lines.append(line)
            id_map[sim_id] = el 
            
        return "\n".join(lines), id_map