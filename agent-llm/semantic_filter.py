# semantic_filter.py

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Set

class SemanticStateFilter:
    """
    Test adımları (step, expected, note) ile UI elementlerini eşleştirir.
    ODS hatalarına (OCR harf düşmesi) karşı dirençlidir.
    """

    # UI'da "kontrol" veya "tıklanabilir nesne" olabilecek kelimeler
    GENERIC_CONTROL_KEYWORDS = {
        "check", "tick", "mark", "box", "toggle", "switch", "radio", 
        "btn", "button", "click", "submit", "ok", "cancel", "apply", 
        "icon", "img", "image", "svg", "path", "vector",
        "input", "edit", "field", "text", "combo", "drop", "list", "select",
        "unknown", "control", "widget", "item", "uiview", "container",
        "m0", "l9", "z", "" 
    }

    # Stop words (bu kelimeler aranmaz)
    STOP_WORDS = {
        "click", "double", "right", "left", "press", "type", "select", 
        "verify", "check", "wait", "button", "input", "list", "panel", 
        "row", "field", "in", "on", "at", "to", "from", "the", "a", "an", "is",
        "find", "locate", "see", "into", "visible", "showing", "options", "should", "be"
    }

    def __init__(self, row_tolerance: int = 15, match_threshold: float = 0.70):
        self.row_tolerance = row_tolerance
        self.match_threshold = match_threshold # %70 benzerlik yeterli (ontro vs Controls)

    def _extract_search_terms(self, text_sources: List[str]) -> Set[str]:
        """Verilen tüm metin kaynaklarından anahtar kelimeleri çıkarır."""
        search_terms = set()
        
        full_text = " ".join([str(t) for t in text_sources if t])
        
        # 1. Tırnak içindeki ifadeler (Highest Priority)
        quoted_phrases = re.findall(r"'(.*?)'", full_text)
        for phrase in quoted_phrases:
            clean_phrase = phrase.strip().lower()
            if len(clean_phrase) > 1:
                search_terms.add(clean_phrase)
                # İfadeyi parçala
                for word in clean_phrase.split():
                    if len(word) > 2:
                        search_terms.add(word)

        # 2. Kelime bazlı temizlik
        clean_text = re.sub(r"[^\w\s]", " ", full_text) # Noktalama kaldır
        words = clean_text.split()
        
        for word in words:
            w_lower = word.lower()
            if w_lower not in self.STOP_WORDS and len(w_lower) > 2:
                search_terms.add(w_lower)
                
        return search_terms

    def _is_fuzzy_match(self, element_name: str, keywords: Set[str]) -> bool:
        """
        Element isminin keywordlerden biriyle eşleşip eşleşmediğini kontrol eder.
        ODS harf düşmelerini (Controls -> Control) yakalar.
        """
        if not element_name:
            return False
            
        name_lower = element_name.lower().strip()
        
        for kw in keywords:
            # 1. Basit içerme kontrolü (Hızlı)
            if kw in name_lower or name_lower in kw:
                return True
                
            # 2. Fuzzy Match (Yavaş ama hataları yakalar)
            # Sadece kelime uzunlukları yakınsa fuzzy match yap (performans için)
            if abs(len(kw) - len(name_lower)) <= 3:
                ratio = SequenceMatcher(None, kw, name_lower).ratio()
                if ratio >= self.match_threshold:
                    return True
                    
        return False

    def _is_potential_control(self, element: Dict[str, Any]) -> bool:
        """Elementin genel bir kontrol (uydu) olup olmadığına bakar."""
        name = str(element.get('name', '')).lower()
        el_type = str(element.get('type', '')).lower()
        
        if len(name) < 2: return True # İsimsiz ikonlar
        
        search_text = f"{name} {el_type}"
        # M0,0L9 gibi SVG pathleri veya generic keywordler
        if any(k in search_text for k in self.GENERIC_CONTROL_KEYWORDS):
            return True
        if re.search(r"m\d+,", name): return True

        return False

    def filter_elements(self, 
                       all_elements: List[Dict[str, Any]], 
                       test_step: str,
                       expected_result: str,
                       note_to_llm: str) -> List[Dict[str, Any]]:
        """
        ANA FİLTRELEME
        State + Step + Expected + Note --> Filtrelenmiş Liste
        """
        # 3 kaynağı birleştirip kelime çıkarıyoruz
        keywords = self._extract_search_terms([test_step, expected_result, note_to_llm])
        
        if not keywords:
            return all_elements[:60] # Fallback

        anchors = []
        anchor_rows = set()
        
        # ADIM 1: Anchor'ları Bul (Fuzzy Match ile)
        for el in all_elements:
            el_name = str(el.get('name', ''))
            
            if self._is_fuzzy_match(el_name, keywords):
                anchors.append(el)
                center = el.get('center', {})
                if center and 'y' in center:
                    anchor_rows.add(int(center['y']))

        # ADIM 2: Satır Genişletme (Row Expansion)
        final_list = list(anchors)
        ids_in_list = {id(el) for el in final_list}
        
        for el in all_elements:
            if id(el) in ids_in_list:
                continue
                
            center = el.get('center', {})
            if not center: continue
            y = int(center.get('y', -999))
            
            # Anchor ile aynı satırda mı?
            is_in_row = any(abs(y - ay) <= self.row_tolerance for ay in anchor_rows)
            
            if is_in_row:
                # Potansiyel kontrol ise ekle (Uydu mantığı)
                if self._is_potential_control(el):
                    final_list.append(el)
                    ids_in_list.add(id(el))
        
        if not final_list:
            return all_elements[:60]
            
        # Orijinal sırayı koruyarak (Y koordinatına göre) sırala ki tablo düzgün görünsün
        final_list.sort(key=lambda e: (e.get('center', {}).get('y', 0), e.get('center', {}).get('x', 0)))
        
        return final_list

    def format_for_llm(self, filtered_elements: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """LLM için optimize edilmiş format: Name başta, Koordinat yanında"""
        
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
            
            # DEĞİŞİKLİK BURADA: ljust metodlarını kaldırdık. 
            # Artık "| Name |" şeklinde sıkışık olacak.
            line = f"{sim_id} | {name} | {etype} | ({x},{y})"
            
            lines.append(line)
            id_map[sim_id] = el 
            
        return "\n".join(lines), id_map