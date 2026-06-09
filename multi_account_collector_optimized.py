#!/usr/bin/env python3
# multi_account_collector_optimized.py
# Multi-account collector ottimizzato per Supabase (cookie dal cron job)

import os
import time
import threading
import signal
import sys
import requests
import cv2
import numpy as np
from datetime import datetime
from supabase import create_client
from datasets import load_dataset

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== CONFIGURAZIONE ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Se non vuoi usare le env var, inseriscili qui direttamente per debug
# SUPABASE_URL = "https://ofijopixtpwahgbwyutc.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9maWpvcGl4dHB3YWhnYnd5dXRjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTkyODIxMiwiZXhwIjoyMDkxNTA0MjEyfQ.BkWb8EuUUJSUUgg3sepDmOdUzsXY7pjGjykQnPMK9q4"

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 5))
STAGGERED_START_DELAY = int(os.environ.get("STAGGERED_START_DELAY", 3))
DIM = 64

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==================== CARICAMENTO DATASET FAISS ====================
def load_faiss_dataset():
    log("📥 Caricamento dataset FAISS...")
    try:
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", trust_remote_code=True)
        data = dataset["train"] if "train" in dataset else dataset
    except Exception as e:
        log(f"⚠️ Errore caricamento dataset: {e}")
        return None, None, None

    X, y, class_to_idx = [], [], {}
    for item in data:
        features = item.get("X")
        label_idx = item.get("y")
        if features is None or label_idx is None:
            continue
        if hasattr(data.features['y'], 'names'):
            class_name = data.features['y'].names[label_idx]
        else:
            class_name = str(label_idx)
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_to_idx)
        X.append(np.array(features, dtype=np.float32))
        y.append(class_to_idx[class_name])

    X_fast = np.vstack(X).astype(np.float32)
    y_fast = np.array(y, dtype=np.int32)
    classes_fast = {v: k for k, v in class_to_idx.items()}
    log(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes_fast)} classi")
    return X_fast, y_fast, classes_fast

# ==================== FUNZIONI PER LE FIGURE ====================
def centra_figura(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])
    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])
    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

def predict_figure(img_crop, X_fast, y_fast, classes_fast):
    if X_fast is None or img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), "errore")

def crop_safe(img, coords):
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

# ==================== FUNZIONI PER SUPABASE ====================
def get_active_cookies():
    """Recupera tutti i cookie attivi da Supabase (tabella account_cookies)"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = supabase.table('account_cookies')\
            .select('nome_utente, divella_format, email, status')\
            .eq('status', 'active')\
            .execute()
        if not resp.data:
            log("⚠️ Nessun cookie attivo trovato in Supabase")
            return []
        accounts = []
        for row in resp.data:
            # Il divella_format è già in formato "nome_utente|cookie_string"
            cookie_str = row['divella_format']
            accounts.append({
                'name': row['nome_utente'],
                'email': row['email'],
                'cookie_string': cookie_str
            })
        log(f"✅ Caricati {len(accounts)} cookie attivi da Supabase")
        return accounts
    except Exception as e:
        log(f"❌ Errore connessione Supabase: {e}")
        return []

def update_cookie_status(email, status):
    """Aggiorna lo stato di un cookie (se scaduto)"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase.table('account_cookies')\
            .update({'status': status, 'updated_at': datetime.now().isoformat()})\
            .eq('email', email)\
            .execute()
        log(f"   📝 Cookie {email} -> {status}")
    except Exception as e:
        log(f"   ❌ Errore aggiornamento: {e}")

# ==================== SURF ACCOUNT ====================
def surf_account(account, X_fast, y_fast, classes_fast):
    account_name = account['name']
    email = account['email']
    cookie_str = account['cookie_string']
    
    # Estrae i cookie dalla stringa (formato: nome|cookie_string)
    # Il cookie_str contiene tutto, lo useremo direttamente negli headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": cookie_str
    }
    session = requests.Session()
    session.headers.update(headers)
    captcha_count = 0
    
    while True:
        try:
            r = session.post("https://www.easyhits4u.com/surf/?ajax=1&try=1", verify=False, timeout=15)
            if r.status_code != 200:
                time.sleep(5)
                continue
            
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap")
            
            if not urlid or not qpic:
                log(f"[{account_name}] ⚠️ Cookie scaduto per {email}")
                update_cookie_status(email, 'expired')
                break
            
            # Captcha a figure
            if picmap and len(picmap) > 0:
                img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
                labels = [predict_figure(c, X_fast, y_fast, classes_fast) for c in crops]
                
                seen = {}
                chosen_idx = None
                for i, label in enumerate(labels):
                    if label and label != "errore":
                        if label in seen:
                            chosen_idx = seen[label]
                            break
                        seen[label] = i
                
                if chosen_idx is None:
                    log(f"[{account_name}] ❌ Nessun duplicato trovato")
                    break
                
                time.sleep(seconds)
                word = picmap[chosen_idx]["value"]
                resp = session.get(
                    f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                    f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                    verify=False
                )
                
                if resp.json().get("warning") == "wrong_choice":
                    log(f"[{account_name}] ❌ Scelta sbagliata")
                    break
                
                captcha_count += 1
                log(f"[{account_name}] ✅ OK #{captcha_count}")
                time.sleep(2)
            
            # Captcha matematico (salva e continua)
            else:
                log(f"[{account_name}] 🧮 Captcha matematico - SALVO")
                # Qui puoi implementare il salvataggio su Supabase Storage
                time.sleep(seconds)
                continue
                
        except Exception as e:
            log(f"[{account_name}] ❌ Errore: {e}")
            time.sleep(5)
            break

# ==================== MAIN ====================
def main():
    global running
    running = True
    
    log("="*60)
    log("🚀 MULTI-ACCOUNT SURF COLLECTOR (Supabase Ottimizzato)")
    log("="*60)
    
    # Carica dataset FAISS
    X_fast, y_fast, classes_fast = load_faiss_dataset()
    if X_fast is None:
        log("❌ Dataset non caricato, esco")
        return
    
    # Recupera cookie da Supabase
    accounts = get_active_cookies()
    if not accounts:
        log("❌ Nessun cookie attivo trovato")
        return
    
    log(f"📋 Account con cookie validi: {len(accounts)}")
    
    # Avvia thread (uno per account)
    threads = []
    for account in accounts:
        while len(threads) >= MAX_CONCURRENT:
            threads = [t for t in threads if t.is_alive()]
            time.sleep(1)
        
        log(f"📧 Avvio: {account['name']} - {account['email']}")
        t = threading.Thread(target=surf_account, args=(account, X_fast, y_fast, classes_fast))
        t.start()
        threads.append(t)
        time.sleep(STAGGERED_START_DELAY)
    
    # Attendi fine
    for t in threads:
        t.join()
    
    log("✅ Raccolta completata!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n🛑 Interrotto dall'utente")
        sys.exit(0)