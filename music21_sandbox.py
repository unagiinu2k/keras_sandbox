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
[type(x) for x in s.getElementsByClass(stream.Stream)] #a lot of "Part"s
[[print(type(y)) for y in x.getElementsByClass(stream.Stream)] for x in s.getElementsByClass(stream.Stream)]
#unlike the deepjazz example, each Parts consists of "Measure"s
type(s)#score

#scoreとPartとMeasureがstreamの基本的なsubclass
#scoreがpartを複数含み、partはmeasureを複数持つ


stream.Score.mro()
s_parts = [x for x in s.parts]

len(s_parts)
s_parts[0].show('midi')
s_parts[1].show('midi')
s_parts_b = [x for x in s.getElementsByClass(stream.Part)]#これでも同じ
s_parts_b[0].show('midi')

measures = [x for x in s_parts[0].getElementsByClass(stream.Measure)]
type(measures)
len(measures)#17小節ある
measures[0]
measures[1]
measures[2]

notes = [x for x in s.getElementsByClass(note.Note)]
len(notes)#zero length
notes = [x for x in s_parts[0].getElementsByClass(note.Note)]
len(notes)#zero length

notes = [x for x in measures[1].getElementsByClass(note.Note)]
len(notes)#zero length

s_parts[0].show()
type(measures)
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

#about offset
#http://web.mit.edu/music21/doc/moduleReference/moduleBase.html#music21.base.Music21Object.offset
import fractions
n1 = note.Note("D#3")
n1.activeSite is None
m1 = stream.Measure() #小節
stream.Measure.mro()
m1.number = 4#小節番号（？）
m1.insert(10.0,n1)
m1.insert(30.0 , note.Note("D3"))
m1.insert(31.0 , note.Note("D3"))
m1.insert(51.0 , note.Note("D3"))#一小節のなかに５２音入っている扱いになるので楽譜の先頭に52/4表記がつく
m1.insert(1.0 , note.Note("B3"))#B3はC3より1オクターブ上
m1.insert(2.0 , note.Note("C3"))
m1.insert(4.0 , note.Note("C3"))
m1.insert(5.0 , note.Note("D3"))
#m1.insert(5.1 , note.Note("E3"))
m1.number
m1.show('midi')
m1.show()#不等間隔の場合ちゃんと表示されない（？）が演奏はちゃんとしてくれる模様（？）

n1.activeSite == m1 #m1が最後に参照されたstreamがm1.activeSite

m2 = stream.Measure()
m2.number = 4
m2.insert(3.0/4  , n1) #４分音符の3/4のところに４分音符が入るので、(3/4+1) * 1/4 = 7/16
n1.offset
m2.show()
m2.show('midi')
#standard hierarchy
#http://web.mit.edu/music21/doc/usersGuide/usersGuide_06_stream2.html
score0 = stream.Score()
part0 = stream.Part()
part0.append([note.Note('F5') , note.Note('F6')])
part0.append([note.Note('F5') , note.Note('F6')])
part0.append([note.Note('F5') , note.Note('F6')])

ks2 = key.KeySignature(-2)#https://matome.naver.jp/odai/2136511911915125501
#key signature(調号)というのは＃やフラットの数だけで完全に決定されるらしい
#なお、key = 調

ks2.sharps
part0.insert(ks2)
part0.insert(note.Note('G5'))
part0.insert(note.Note('G8'))#ものすごく高い音を加える
part0.insert(3,note.Note('G2'))#
part0.insert(1,note.Note('G1'))#
for x in part0:
    print(x.offset)

#part0.insert(-5,note.Note('G1'))#
part0.show('midi')
part0.show()
part0.measures(0,1).show() #measure = 小節　 specify semi-open period
m00 = part0.measures(0,1)
type(m00)

score0.insert(0,part0)
part1 = stream.Part()
part1.append([note.Note('F3') , note.Note('F4')])

score0.insert(20,part1)
score0.show()
score0.show('midi')
score0.analyze('key')
for el in score0.flat:
    print(el.offset ,  el , el.activeSite)

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
f2.pitch.frequency #pitch=音の高さ
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