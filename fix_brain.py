"""
Brain Reset Utility - Fixes corrupted brain files
"""

import os
import sys

BRAIN_FILE = 'my_brain.pkl'

print("="*70)
print("BRAIN RESET UTILITY")
print("="*70)
print()

# Check if brain file exists
if os.path.exists(BRAIN_FILE):
    print(f"Found brain file: {BRAIN_FILE}")
    print(f"Size: {os.path.getsize(BRAIN_FILE)} bytes")

    # Try to load it
    print("\nAttempting to load brain...")
    try:
        from brain_io import load_brain
        brain = load_brain(BRAIN_FILE)
        print("\n✅ Brain is healthy and loaded successfully!")
        print(f"   Neurons: {brain.num_neurons}")
        print(f"   Input channels: {brain.input_channels}")

    except Exception as e:
        print(f"\n❌ Brain is corrupted!")
        print(f"   Error: {e}")

        # Ask to reset
        print("\n" + "="*70)
        response = input("Would you like to delete the corrupted brain and create a new one? (y/n): ")

        if response.lower() == 'y':
            # Backup corrupted file
            backup_file = BRAIN_FILE + '.corrupted.bak'
            try:
                import shutil
                shutil.copy2(BRAIN_FILE, backup_file)
                print(f"\n✓ Backed up corrupted brain to: {backup_file}")
            except Exception as backup_err:
                print(f"\n⚠️  Could not create backup: {backup_err}")

            # Delete corrupted file
            try:
                os.remove(BRAIN_FILE)
                print(f"✓ Deleted corrupted brain file")
            except Exception as del_err:
                print(f"❌ Could not delete file: {del_err}")
                sys.exit(1)

            # Create new brain
            print("\nCreating new brain...")
            try:
                from brain_io import load_brain
                new_brain = load_brain(BRAIN_FILE)
                print("\n✅ New brain created successfully!")
                print(f"   Neurons: {new_brain.num_neurons}")
                print(f"   Input channels: {new_brain.input_channels}")

                # Save it
                from brain_io import save_brain
                save_brain(new_brain, BRAIN_FILE)
                print(f"\n✓ Saved new brain to: {BRAIN_FILE}")

            except Exception as create_err:
                print(f"\n❌ Could not create new brain: {create_err}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print("\nNo changes made.")
            sys.exit(0)
else:
    print(f"No brain file found at: {BRAIN_FILE}")
    print("\nCreating new brain...")

    try:
        from brain_io import load_brain, save_brain
        new_brain = load_brain(BRAIN_FILE)
        save_brain(new_brain, BRAIN_FILE)
        print("\n✅ New brain created and saved!")
        print(f"   Neurons: {new_brain.num_neurons}")
        print(f"   Input channels: {new_brain.input_channels}")
        print(f"   File: {BRAIN_FILE}")
    except Exception as e:
        print(f"\n❌ Could not create brain: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*70)
print("✅ BRAIN READY!")
print("="*70)
print("\nYou can now run the web dashboard:")
print("  launch_dashboard.bat")
print("\nOr use the command:")
print("  streamlit run web_app.py")
print()

