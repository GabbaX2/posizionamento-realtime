#!/usr/bin/env python3
"""
Script per compilare TypeScript e avviare l'applicazione
"""

import os
import subprocess
import sys


def build_typescript():
    """Compila il TypeScript"""
    print("🔨 Compilazione TypeScript...")

    # Crea directory src se non esiste
    os.makedirs('src', exist_ok=True)

    # Verifica che main.ts esista
    if not os.path.exists('src/main.ts'):
        print("❌ File src/main.ts non trovato!")
        print("📝 Creazione del file main.ts...")

        # Crea un file main.ts di base
        main_ts_content = '''// File principale TypeScript
// Incolla qui il codice TypeScript che ti è stato fornito

console.log('TypeScript file creato. Sostituisci con il codice completo.');
'''
        with open('src/main.ts', 'w') as f:
            f.write(main_ts_content)

    # Installa TypeScript se necessario
    if not os.path.exists('node_modules'):
        print("📦 Installazione TypeScript...")
        subprocess.run(['npm', 'install'], check=True)

    # Compila TypeScript
    try:
        result = subprocess.run(['npm', 'run', 'build'], check=True, capture_output=True, text=True)
        print("✅ TypeScript compilato con successo")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore nella compilazione TypeScript: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


def main():
    """Funzione principale"""
    print("🚀 Sensor Placement Validator - Build System")
    print("=" * 50)

    # Compila TypeScript
    if build_typescript():
        print("\n🎉 Build completata!")
        print("\nPer avviare l'applicazione:")
        print("  python app.py")
        print("\nOppure per sviluppo:")
        print("  npm run watch  # Compilazione automatica")
        print("  python app.py  # In altro terminale")
    else:
        print("\n❌ Build fallita!")
        sys.exit(1)


if __name__ == '__main__':
    main()