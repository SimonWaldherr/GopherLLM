package wordpiece

// BertNormalizer strips accents by applying canonical decomposition (NFD)
// and then dropping non-spacing marks. Keeping the canonical single-rune
// decompositions as a generated table preserves the module's no-third-party
// dependency policy while covering scripts beyond the Latin-1 subset.
//
// The two strings are parallel, code-point-sorted Unicode decomposition data:
// each rune in wordPieceNFDComposed maps to the rune at the same position in
// wordPieceNFDBaseRunes. Combining marks already present in input are removed
// directly by the public tokenizer's WordPiece preprocessing.
const wordPieceNFDComposed = "ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝàáâãäåçèéêëìíîïñòóôõöùúûüýÿĀāĂăĄąĆćĈĉĊċČčĎďĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĨĩĪīĬĭĮįİĴĵĶķĹĺĻļĽľŃńŅņŇňŌōŎŏŐőŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžƠơƯưǍǎǏǐǑǒǓǔǕǖǗǘǙǚǛǜǞǟǠǡǢǣǦǧǨǩǪǫǬǭǮǯǰǴǵǸǹǺǻǼǽǾǿȀȁȂȃȄȅȆȇȈȉȊȋȌȍȎȏȐȑȒȓȔȕȖȗȘșȚțȞȟȦȧȨȩȪȫȬȭȮȯȰȱȲȳ̈́΅ΆΈΉΊΌΎΏΐΪΫάέήίΰϊϋόύώϓϔЀЁЃЇЌЍЎЙйѐёѓїќѝўѶѷӁӂӐӑӒӓӖӗӚӛӜӝӞӟӢӣӤӥӦӧӪӫӬӭӮӯӰӱӲӳӴӵӸӹآأؤإئۀۂۓऩऱऴक़ख़ग़ज़ड़ढ़फ़य़ড়ঢ়য়ਲ਼ਸ਼ਖ਼ਗ਼ਜ਼ਫ਼ୈଡ଼ଢ଼ైේགྷཌྷདྷབྷཛྷཀྵཱཱིུྲྀླཱྀྀྒྷྜྷྡྷྦྷྫྷྐྵဦḀḁḂḃḄḅḆḇḈḉḊḋḌḍḎḏḐḑḒḓḔḕḖḗḘḙḚḛḜḝḞḟḠḡḢḣḤḥḦḧḨḩḪḫḬḭḮḯḰḱḲḳḴḵḶḷḸḹḺḻḼḽḾḿṀṁṂṃṄṅṆṇṈṉṊṋṌṍṎṏṐṑṒṓṔṕṖṗṘṙṚṛṜṝṞṟṠṡṢṣṤṥṦṧṨṩṪṫṬṭṮṯṰṱṲṳṴṵṶṷṸṹṺṻṼṽṾṿẀẁẂẃẄẅẆẇẈẉẊẋẌẍẎẏẐẑẒẓẔẕẖẗẘẙẛẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹἀἁἂἃἄἅἆἇἈἉἊἋἌἍἎἏἐἑἒἓἔἕἘἙἚἛἜἝἠἡἢἣἤἥἦἧἨἩἪἫἬἭἮἯἰἱἲἳἴἵἶἷἸἹἺἻἼἽἾἿὀὁὂὃὄὅὈὉὊὋὌὍὐὑὒὓὔὕὖὗὙὛὝὟὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯὰάὲέὴήὶίὸόὺύὼώᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍᾎᾏᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝᾞᾟᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯᾰᾱᾲᾳᾴᾶᾷᾸᾹᾺΆᾼ῁ῂῃῄῆῇῈΈῊΉῌ῍῎῏ῐῑῒΐῖῗῘῙῚΊ῝῞῟ῠῡῢΰῤῥῦῧῨῩῪΎῬ῭΅ῲῳῴῶῷῸΌῺΏῼÅ↚↛↮⇍⇎⇏∄∉∌∤∦≁≄≇≉≠≢≭≮≯≰≱≴≵≸≹⊀⊁⊄⊅⊈⊉⊬⊭⊮⊯⋠⋡⋢⋣⋪⋫⋬⋭⫝̸がぎぐげござじずぜぞだぢづでどばぱびぴぶぷべぺぼぽゔゞガギグゲゴザジズゼゾダヂヅデドバパビピブプベペボポヴヷヸヹヺヾיִײַשׁשׂשּׁשּׂאַאָאּבּגּדּהּוּזּטּיּךּכּלּמּנּסּףּפּצּקּרּשּתּוֹבֿכֿפֿ𐗉𐗤𑂚𑂜𑂫𑄮𑄯𑎅𑒻𖄡𖄢𖄣𖄤𖄥𖄦𖄧𖄨"

const wordPieceNFDBaseRunes = "AAAAAACEEEEIIIINOOOOOUUUUYaaaaaaceeeeiiiinooooouuuuyyAaAaAaCcCcCcCcDdEeEeEeEeEeGgGgGgGgHhIiIiIiIiIJjKkLlLlLlNnNnNnOoOoOoRrRrRrSsSsSsSsTtTtUuUuUuUuUuUuWwYyYZzZzZzOoUuAaIiOoUuUuUuUuUuAaAaÆæGgKkOoOoƷʒjGgNnAaÆæØøAaAaEeEeIiIiOoOoRrRrUuUuSsTtHhAaEeOoOoOoOoYÿ¨ΑΕΗΙΟΥΩιΙΥαεηιυιυουωϒϒЕЕГІКИУИиеегікиуѴѵЖжАаАаЕеӘәЖжЗзИиИиОоӨөЭэУуУуУуЧчЫыااوايەہےनरळकखगजडढफयডঢযਲਸਖਗਜਫେଡଢెෙགཌདབཛཀཱཱྲླཱྒྜྡྦྫྐဥAaBbBbBbCcDdDdDdDdDdEeEeEeEeEeFfGgHhHhHhHhHhIiIiKkKkKkLlLlLlLlMmMmMmNnNnNnNnOoOoOoOoPpPpRrRrRrRrSsSsSsSsSsTtTtTtTtUuUuUuUuUuVvVvWwWwWwWwWwXxXxYyZzZzZzhtwyſAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYyααααααααΑΑΑΑΑΑΑΑεεεεεεΕΕΕΕΕΕηηηηηηηηΗΗΗΗΗΗΗΗιιιιιιιιΙΙΙΙΙΙΙΙοοοοοοΟΟΟΟΟΟυυυυυυυυΥΥΥΥωωωωωωωωΩΩΩΩΩΩΩΩααεεηηιιοουυωωααααααααΑΑΑΑΑΑΑΑηηηηηηηηΗΗΗΗΗΗΗΗωωωωωωωωΩΩΩΩΩΩΩΩαααααααΑΑΑΑΑ¨ηηηηηΕΕΗΗΗ᾿᾿᾿ιιιιιιΙΙΙΙ῾῾῾υυυυρρυυΥΥΥΥΡ¨¨ωωωωωΟΟΩΩΩA←→↔⇐⇔⇒∃∈∋∣∥∼≃≅≈=≡≍<>≤≥≲≳≶≷≺≻⊂⊃⊆⊇⊢⊨⊩⊫≼≽⊑⊒⊲⊳⊴⊵⫝かきくけこさしすせそたちつてとははひひふふへへほほうゝカキクケコサシスセソタチツテトハハヒヒフフヘヘホホウワヰヱヲヽיײששששאאאבגדהוזטיךכלמנסףפצקרשתובכפ𐗒𐗚𑂙𑂛𑂥𑄱𑄲𑎄𑒹𖄡𖄢𖄡𖄡"

var wordPieceNFDBase = func() map[rune]rune {
	composed := []rune(wordPieceNFDComposed)
	bases := []rune(wordPieceNFDBaseRunes)
	if len(composed) != len(bases) {
		panic("invalid WordPiece NFD table")
	}
	table := make(map[rune]rune, len(composed))
	for i, r := range composed {
		table[r] = bases[i]
	}
	return table
}()

// StripAccent returns the canonical base rune used by BERT normalization.
func StripAccent(r rune) rune {
	if base, ok := wordPieceNFDBase[r]; ok {
		return base
	}
	return r
}
