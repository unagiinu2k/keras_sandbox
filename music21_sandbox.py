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

converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#").show()
s = converter.parse('/home/naraba/program/python/music21/etude10-05.xml')
s.show()

f =note.Note("F5")
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