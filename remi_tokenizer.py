import miditoolkit
import math

class Vocabulary:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        self.add_token(self.pad_token)
        self.add_token(self.bos_token)
        self.add_token(self.eos_token)
        
    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            
    def encode(self, tokens):
        return [self.token2id.get(t, self.token2id[self.pad_token]) for t in tokens]
        
    def decode(self, ids):
        return [self.id2token.get(i, self.pad_token) for i in ids]
        
    def __len__(self):
        return len(self.token2id)
        
    @property
    def pad_id(self):
        return self.token2id[self.pad_token]
        
    @property
    def bos_id(self):
        return self.token2id[self.bos_token]
        
    @property
    def eos_id(self):
        return self.token2id[self.eos_token]


class REMITokenizer:
    def __init__(self):
        self.vocab = Vocabulary()
        
        self.pitch_range = range(21, 109)
        self.velocity_bins = 32
        self.duration_bins = 64
        self.position_resolution = 16
        
        self._build_vocab()
        
    def _build_vocab(self):
        # Bar token
        self.vocab.add_token("Bar")
        
        # Position tokens (0 to 15)
        for i in range(self.position_resolution):
            self.vocab.add_token(f"Position_{i}")
            
        # Pitch tokens
        for p in self.pitch_range:
            self.vocab.add_token(f"Pitch_{p}")
            
        # Velocity tokens
        for v in range(self.velocity_bins):
            self.vocab.add_token(f"Velocity_{v}")
            
        # Duration tokens
        for d in range(1, self.duration_bins + 1):
            self.vocab.add_token(f"Duration_{d}")
            
    def _quantize_velocity(self, velocity):
        return min(math.floor((velocity / 128.0) * self.velocity_bins), self.velocity_bins - 1)
        
    def _dequantize_velocity(self, bin_idx):
        return int((bin_idx + 0.5) * (128.0 / self.velocity_bins))
        
    def _quantize_duration(self, duration_ticks, ticks_per_quarter):
        position_ticks = ticks_per_quarter / self.position_resolution
        duration_positions = max(1, round(duration_ticks / position_ticks))
        return min(duration_positions, self.duration_bins)
        
    def _dequantize_duration(self, bin_idx, ticks_per_quarter):
        position_ticks = ticks_per_quarter / self.position_resolution
        return int(bin_idx * position_ticks)

    def midi_to_tokens(self, midi_path):
        midi = miditoolkit.midi.parser.MidiFile(midi_path)
        ticks_per_quarter = midi.ticks_per_beat
        ticks_per_bar = ticks_per_quarter * 4
        
        notes = []
        for inst in midi.instruments:
            notes.extend(inst.notes)
            
        # Sort notes by start time, then pitch
        notes.sort(key=lambda x: (x.start, x.pitch))
        
        tokens = []
        current_bar = -1
        
        for note in notes:
            if not (21 <= note.pitch <= 108):
                continue
                
            bar = note.start // ticks_per_bar
            
            if bar > current_bar:
                for b in range(current_bar + 1, bar + 1):
                    tokens.append("Bar")
                current_bar = bar
                
            tick_in_bar = note.start % ticks_per_bar
            position = round(tick_in_bar / (ticks_per_quarter / self.position_resolution))
            position = min(max(position, 0), self.position_resolution - 1)
            
            pitch = note.pitch
            velocity = self._quantize_velocity(note.velocity)
            duration = self._quantize_duration(note.end - note.start, ticks_per_quarter)
            
            tokens.append(f"Position_{position}")
            tokens.append(f"Pitch_{pitch}")
            tokens.append(f"Velocity_{velocity}")
            tokens.append(f"Duration_{duration}")
            
        return tokens
        
    def tokens_to_midi(self, tokens, output_path):
        midi = miditoolkit.midi.parser.MidiFile()
        ticks_per_quarter = midi.ticks_per_beat
        ticks_per_bar = ticks_per_quarter * 4
        
        inst = miditoolkit.midi.containers.Instrument(program=0, is_drum=False, name="Piano")
        
        current_bar = -1
        current_position = 0
        current_pitch = -1
        current_velocity = -1
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == "Bar":
                current_bar += 1
            elif token.startswith("Position_"):
                current_position = int(token.split("_")[1])
            elif token.startswith("Pitch_"):
                current_pitch = int(token.split("_")[1])
            elif token.startswith("Velocity_"):
                current_velocity = int(token.split("_")[1])
            elif token.startswith("Duration_"):
                duration_bin = int(token.split("_")[1])
                
                if current_bar >= 0 and current_pitch != -1 and current_velocity != -1:
                    start_tick = current_bar * ticks_per_bar + int(current_position * (ticks_per_quarter / self.position_resolution))
                    duration_ticks = self._dequantize_duration(duration_bin, ticks_per_quarter)
                    end_tick = start_tick + duration_ticks
                    velocity = self._dequantize_velocity(current_velocity)
                    
                    note = miditoolkit.midi.containers.Note(
                        velocity=velocity,
                        pitch=current_pitch,
                        start=start_tick,
                        end=end_tick
                    )
                    inst.notes.append(note)
                    
                current_pitch = -1
                current_velocity = -1
            
            i += 1
            
        midi.instruments.append(inst)
        midi.dump(output_path)
