# Antares
Antares is a chess engine written in python and partially JIT compiled using Numba. My goal is to make the strongest traditional (minimax) python engine,
and I aim for this engine to reach a rating of 2200. Antares uses the UCI protocol.

## Download
There is a release for windows in the dist folder. *NOTE: Antares doesn't work with Cutechess due to long initialization times. When first opening Antares,
it takes around 20-45 seconds to initialize, meaning responding with readyok after receiving uci isready command. Subsequently, it will take 10-15 seconds to
initialize as functions are cached. The reason for this initialization time is due to using Numba, a JIT compiler.*
## Lichess
You can play Antares [here](https://lichess.org/@/AntaresPy)

## Board Representation
Antares uses a 10x12 mailbox array. This board representation gives padding around the 8x8 board for faster move generation. Antares also features
piece lists.

## Move Generation
Offsets / Increments are used in loops to generate the pseudo-legal moves. Within the search and perft, the moves are tested for 
legality, and if the move is not legal, we skip to the next move.

## Search
#### Iterative Deepening
- Aspiration Windows
  - Negamax (Minimax)
    - Alpha-Beta Pruning
      - Null Move Pruning
      - Principal Variation Search
      - Late Move Reductions
  - Quiescence Search
  
#### Move Ordering
- Transposition Table Move
- Captures
  - Most Valuable Victim - Least Valuable Agressor (MVV-LVA)
- Killer Moves
- History Moves
- Promotions
- Castling
- Piece Square Tables (Value of target square - Value of origin square)

## Evaluation
- Material
- Piece Square Tables
- Double, Isolated, Backwards, and Passed Pawns
- Open and semi-open files for rooks and kings
- King's pawn-shield
- Bishop Pair
- Tempo
- Tapered evaluation between the middlegame and endgame

## Credit and Thanks
- [The Chess Programming Wiki](https://www.chessprogramming.org/Main_Page) contains invaluable resources that I have used.
- [Bitboard Chess Engine in C](https://www.youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs) A nice series that contains implementations
for different ideas and features.
- [Sunfish Engine](https://github.com/thomasahle/sunfish) A great resource.
- [Black Numba Engine](https://github.com/Avo-k/black_numba) Another great resource for learning how to use Numba with a chess engine.

