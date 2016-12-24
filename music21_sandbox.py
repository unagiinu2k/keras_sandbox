#http://d.hatena.ne.jp/naraba/20121201/p1
#http://web.mit.edu/music21/doc/usersGuide/usersGuide_01_installing.html

from music21 import *
environment.set('musicxmlPath' , r"C:\Program Files (x86)\Finale NotePad 2012\Finale NotePad.exe")
#configure.run()
#environment.keys()
#environment.get('musicxmlPath')

s = corpus.parse('bach/bwv65.2.xml')
s.analyze('key')
s.show('midi')
type(s)
s_parts = [x for x in s.parts]
len(s_parts)
s_parts[0].show('midi')
s.show()

p = converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
q = converter.parse("tinynotation: 4/4 c4 DD4 e''4 f1# g8b h4")
type(converter.parse("tinynotation: 4/4 c4 DD4 e''4 f1# g8b h4"))
type(p)
#s = converter.parse('/home/naraba/program/python/music21/etude10-05.xml')

stream.Stream.mro()
stream.Part.mro()
s2 = stream.Stream()
s2.insert(0  , p)#adding part, first argument should be offset??
s2.insert(320  , q)
s2.show()
s2.show('midi')
s2_parts = [x for x in s2.parts]#error

#standard hierarchy
#http://web.mit.edu/music21/doc/usersGuide/usersGuide_06_stream2.html
score0 = stream.Score()
part0 = stream.Part()
part0.append([note.Note('F5') , note.Note('F6')])
part0.show('midi')
part0.measures(0,1).show() #measure = 小節　 specify semi-open period
score0.insert(0,part0)
part1 = stream.Part()
part1.append([note.Note('F5') , note.Note('F6')])

score0.insert(320,part1)
score0.show()
len(score0.getElementsByClass(stream.Part))
len(score0.getElementsByClass(note.Note)) # returns zero as expected
run_score.insert(0 , p)
run_score.insert(320 , q)
type(p)
run_score_parts = [x for x in run_score]#error
len(run_score_parts)
from music21 import chord
c1 = chord.Chord(["C4" , "G4" , "G4"]) #和音
chord.Chord.mro()
c1.show('midi')
c1.show()
f5 = note.Note("F5")
f.name
f.octave
f.pitch.frequency
f.pitch.pitchClass
f.pitch.accidental
f.duration.quarterLength = 3
f.duration
f.step
f2 = note.Note("F#5")
f2.pitch.frequency
f2.pitch.pitchClass
b_5 = note.Note("b-5")
b_5.pitch.frequency
b_5.pitch.accidental.displayLocation
b_5.tmp = "test"
b_5.tmp
type(b_5)

b_5.pitch.pitchClass
b_5.transpose("M3").pitch.pitchClass
b_5.transpose("d6").pitch.pitchClass
c5 = note.Note("c5")
c5.transpose("M3").name
c5.transpose("d6").name
c5.pitch.accidental.name
r = note.Rest("whole")
r.show()

noteList = [f , f2, b_5]
noteList.append(c5)
print(noteList)

stream0 = stream.Stream()
stream0.append(noteList)
stream0.show()
for p in stream0:
    print(p.step) #step は　「音程」
[print(p.step) for p in stream0] #bad

stream0.getElementsByClass("note.Note")
stream0.analyze('ambitus')# 音域
stream1 = stream.Stream([note.Note("F5"),note.Note("F#6")])
stream1 = stream.Stream([note.Note("F5"),note.Note("G5")])
stream1.analyze("ambitus")
stream1.append(b_5)
stream1[1].offset
for n in stream1:
    print(n.offset)#時間方向のoffset(?)

stream1.append(note.Note("F6"))
stream1.append(note.Note("F8"))
stream1.show()

from music21 import stream
stream.Voice

m1 = stream1.getElementsByClass(stream.Voice)
len(m1)